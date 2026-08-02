use std::convert::TryFrom;
use std::fs::{self, File};
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};

const ELF64_HEADER_SIZE: usize = 64;
const ELF64_PROGRAM_HEADER_SIZE: usize = 56;
const ELF64_SECTION_HEADER_SIZE: usize = 64;
const ELF64_DYNAMIC_ENTRY_SIZE: usize = 16;
const ELF64_SYMBOL_SIZE: usize = 24;
const MAX_ELF_FILE_SIZE: u64 = 128 * 1024 * 1024;
const MAX_PROGRAM_HEADERS: usize = 4_096;
const MAX_SECTION_HEADERS: usize = 65_534;
const MAX_DYNAMIC_ENTRIES: usize = 65_536;
const MAX_DYNAMIC_SYMBOLS: usize = 1_000_000;
const MAX_DYNAMIC_STRING_TABLE: usize = 32 * 1024 * 1024;
const MAX_HASH_BUCKETS: usize = 1_000_000;

// curl 0.4.50's always-built safe wrapper references the easy and multi
// entry points below. Requiring these definitions makes the direct system
// library contract explicit instead of accepting an arbitrary ET_DYN file
// whose basename happens to be libcurl.so.4.
const REQUIRED_CURL_SYMBOLS: &[&[u8]] = &[
    b"curl_global_init",
    b"curl_global_cleanup",
    b"curl_version",
    b"curl_version_info",
    b"curl_easy_init",
    b"curl_easy_cleanup",
    b"curl_easy_setopt",
    b"curl_easy_perform",
    b"curl_easy_getinfo",
    b"curl_easy_strerror",
    b"curl_slist_append",
    b"curl_slist_free_all",
    b"curl_multi_init",
    b"curl_multi_cleanup",
    b"curl_multi_add_handle",
    b"curl_multi_remove_handle",
    b"curl_multi_perform",
    b"curl_multi_info_read",
    b"curl_multi_wait",
    b"curl_multi_timeout",
    b"curl_multi_setopt",
    b"curl_multi_socket_action",
    b"curl_multi_strerror",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinuxMachine {
    X86_64,
    Aarch64,
}

impl LinuxMachine {
    fn elf_machine(self) -> u16 {
        match self {
            Self::X86_64 => 62,
            Self::Aarch64 => 183,
        }
    }

    fn multiarch(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64-linux-gnu",
            Self::Aarch64 => "aarch64-linux-gnu",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Target {
    MacArm64,
    Linux(LinuxMachine),
}

pub(crate) fn ensure_native_build(host: &str, target: &str) -> Result<(), String> {
    if host == target {
        return Ok(());
    }
    Err(format!(
        "deltafin's direct curl-sys fork supports native builds only; HOST {host:?} does not match TARGET {target:?}"
    ))
}

pub(crate) fn parse_supported_target(target: &str) -> Result<Target, String> {
    match target {
        "aarch64-apple-darwin" => Ok(Target::MacArm64),
        "x86_64-unknown-linux-gnu" => Ok(Target::Linux(LinuxMachine::X86_64)),
        "aarch64-unknown-linux-gnu" => Ok(Target::Linux(LinuxMachine::Aarch64)),
        _ => Err(format!(
            "deltafin's direct curl-sys fork supports only aarch64-apple-darwin, \
             x86_64-unknown-linux-gnu, and aarch64-unknown-linux-gnu; got {target:?}"
        )),
    }
}

pub(crate) fn ensure_supported_features(features: &[(&str, bool)]) -> Result<(), String> {
    let enabled: Vec<&str> = features
        .iter()
        .filter_map(|(name, enabled)| enabled.then_some(*name))
        .collect();
    if enabled.is_empty() {
        return Ok(());
    }
    Err(format!(
        "unsupported curl-sys feature(s) for the direct system-libcurl build: {}. \
         This fork accepts only the `ssl` system-TLS marker and \
         `force-system-lib-on-osx`; it never builds or reconfigures libcurl",
        enabled.join(", ")
    ))
}

fn standard_library_dirs(machine: LinuxMachine) -> Vec<PathBuf> {
    let multiarch = machine.multiarch();
    Vec::from([
        format!("/usr/lib/{multiarch}"),
        format!("/lib/{multiarch}"),
        "/usr/lib64".to_owned(),
        "/lib64".to_owned(),
        "/usr/lib".to_owned(),
        "/lib".to_owned(),
    ])
    .into_iter()
    .map(PathBuf::from)
    .collect()
}

#[derive(Clone, Copy, Debug)]
struct ElfHeader {
    program_offset: u64,
    program_count: usize,
    section_offset: u64,
    section_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct LoadSegment {
    file_offset: u64,
    virtual_address: u64,
    file_size: u64,
}

#[derive(Clone, Copy, Debug)]
struct DynamicSegment {
    file_offset: u64,
    virtual_address: u64,
    file_size: u64,
}

#[derive(Clone, Debug)]
struct NamedRange {
    name: &'static str,
    range: Range<usize>,
}

#[derive(Default)]
struct DynamicInfo {
    string_table: Option<u64>,
    string_size: Option<u64>,
    symbol_table: Option<u64>,
    symbol_entry_size: Option<u64>,
    soname: Option<u64>,
    sysv_hash: Option<u64>,
    gnu_hash: Option<u64>,
    referenced_strings: Vec<(&'static str, u64)>,
}

fn checked_range(
    offset: u64,
    size: u64,
    file_len: usize,
    label: &str,
) -> Result<Range<usize>, String> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| format!("{label} range overflows u64"))?;
    let start = usize::try_from(offset)
        .map_err(|_| format!("{label} offset does not fit this build host"))?;
    let end =
        usize::try_from(end).map_err(|_| format!("{label} end does not fit this build host"))?;
    if end > file_len {
        return Err(format!(
            "{label} range {start}..{end} exceeds file length {file_len}"
        ));
    }
    Ok(start..end)
}

fn read_u16(bytes: &[u8], offset: usize, label: &str) -> Result<u16, String> {
    let range = checked_range(offset as u64, 2, bytes.len(), label)?;
    Ok(u16::from_le_bytes([
        bytes[range.start],
        bytes[range.start + 1],
    ]))
}

fn read_u32(bytes: &[u8], offset: usize, label: &str) -> Result<u32, String> {
    let range = checked_range(offset as u64, 4, bytes.len(), label)?;
    Ok(u32::from_le_bytes([
        bytes[range.start],
        bytes[range.start + 1],
        bytes[range.start + 2],
        bytes[range.start + 3],
    ]))
}

fn read_u64(bytes: &[u8], offset: usize, label: &str) -> Result<u64, String> {
    let range = checked_range(offset as u64, 8, bytes.len(), label)?;
    Ok(u64::from_le_bytes([
        bytes[range.start],
        bytes[range.start + 1],
        bytes[range.start + 2],
        bytes[range.start + 3],
        bytes[range.start + 4],
        bytes[range.start + 5],
        bytes[range.start + 6],
        bytes[range.start + 7],
    ]))
}

fn parse_elf64_header(bytes: &[u8], expected_machine: LinuxMachine) -> Result<ElfHeader, String> {
    if bytes.len() < ELF64_HEADER_SIZE {
        return Err("file is shorter than an ELF64 header".to_owned());
    }
    if &bytes[0..4] != b"\x7fELF" {
        return Err("file does not have ELF magic".to_owned());
    }
    if bytes[4] != 2 {
        return Err(format!("ELF class {} is not ELF64", bytes[4]));
    }
    if bytes[5] != 1 {
        return Err(format!(
            "ELF data encoding {} is not little-endian",
            bytes[5]
        ));
    }
    if bytes[6] != 1 {
        return Err(format!(
            "ELF identification version {} is invalid",
            bytes[6]
        ));
    }
    if !matches!(bytes[7], 0 | 3) {
        return Err(format!(
            "ELF OS ABI {} is neither System V nor GNU/Linux",
            bytes[7]
        ));
    }
    if bytes[8] != 0 {
        return Err(format!("ELF ABI version {} is unsupported", bytes[8]));
    }
    if bytes[9..16].iter().any(|byte| *byte != 0) {
        return Err("ELF identification padding is not zeroed".to_owned());
    }

    let object_type = read_u16(bytes, 16, "ELF e_type")?;
    if object_type != 3 {
        return Err(format!("ELF object type {object_type} is not ET_DYN"));
    }
    let machine = read_u16(bytes, 18, "ELF e_machine")?;
    if machine != expected_machine.elf_machine() {
        return Err(format!(
            "ELF e_machine {machine} does not match expected {}",
            expected_machine.elf_machine()
        ));
    }
    if read_u32(bytes, 20, "ELF e_version")? != 1 {
        return Err("ELF e_version is not EV_CURRENT".to_owned());
    }
    if read_u16(bytes, 52, "ELF e_ehsize")? as usize != ELF64_HEADER_SIZE {
        return Err("ELF e_ehsize is not 64".to_owned());
    }

    let program_offset = read_u64(bytes, 32, "ELF e_phoff")?;
    let program_entry_size = read_u16(bytes, 54, "ELF e_phentsize")? as usize;
    let program_count = read_u16(bytes, 56, "ELF e_phnum")? as usize;
    if program_entry_size != ELF64_PROGRAM_HEADER_SIZE {
        return Err(format!(
            "ELF program-header entry size {program_entry_size} is not {ELF64_PROGRAM_HEADER_SIZE}"
        ));
    }
    if program_count == 0 || program_count > MAX_PROGRAM_HEADERS {
        return Err(format!(
            "ELF program-header count {program_count} is outside 1..={MAX_PROGRAM_HEADERS}"
        ));
    }

    let section_offset = read_u64(bytes, 40, "ELF e_shoff")?;
    let section_entry_size = read_u16(bytes, 58, "ELF e_shentsize")? as usize;
    let section_count = read_u16(bytes, 60, "ELF e_shnum")? as usize;
    let section_string_index = read_u16(bytes, 62, "ELF e_shstrndx")? as usize;
    if section_offset == 0 {
        if section_count != 0 || section_string_index != 0 {
            return Err("ELF has section counts but no section-header table".to_owned());
        }
        if !matches!(section_entry_size, 0 | ELF64_SECTION_HEADER_SIZE) {
            return Err(format!(
                "ELF empty section-header entry size {section_entry_size} is invalid"
            ));
        }
    } else {
        if section_entry_size != ELF64_SECTION_HEADER_SIZE {
            return Err(format!(
                "ELF section-header entry size {section_entry_size} is not {ELF64_SECTION_HEADER_SIZE}"
            ));
        }
        if section_count == 0 || section_count > MAX_SECTION_HEADERS {
            return Err(format!(
                "ELF section-header count {section_count} is outside 1..={MAX_SECTION_HEADERS}; extended numbering is unsupported"
            ));
        }
        if section_string_index != 0 && section_string_index >= section_count {
            return Err(format!(
                "ELF section-string index {section_string_index} is outside {section_count} sections"
            ));
        }
    }

    Ok(ElfHeader {
        program_offset,
        program_count,
        section_offset,
        section_count,
    })
}

#[cfg(test)]
pub(crate) fn validate_elf64_header(
    header: &[u8],
    expected_machine: LinuxMachine,
) -> Result<(), String> {
    parse_elf64_header(header, expected_machine).map(|_| ())
}

fn table_range(
    offset: u64,
    count: usize,
    entry_size: usize,
    file_len: usize,
    label: &str,
) -> Result<Range<usize>, String> {
    let size = count
        .checked_mul(entry_size)
        .ok_or_else(|| format!("{label} byte count overflows usize"))?;
    checked_range(offset, size as u64, file_len, label)
}

fn reject_overlapping_ranges(ranges: &[NamedRange]) -> Result<(), String> {
    for (index, left) in ranges.iter().enumerate() {
        for right in &ranges[index + 1..] {
            if left.range.start < right.range.end && right.range.start < left.range.end {
                return Err(format!(
                    "ELF structures {} ({}..{}) and {} ({}..{}) overlap",
                    left.name,
                    left.range.start,
                    left.range.end,
                    right.name,
                    right.range.start,
                    right.range.end
                ));
            }
        }
    }
    Ok(())
}

fn map_virtual_range(
    loads: &[LoadSegment],
    virtual_address: u64,
    size: u64,
    file_len: usize,
    label: &str,
) -> Result<Range<usize>, String> {
    let virtual_end = virtual_address
        .checked_add(size)
        .ok_or_else(|| format!("{label} virtual range overflows u64"))?;
    let mut mapped: Option<Range<usize>> = None;
    for load in loads {
        let load_end = load
            .virtual_address
            .checked_add(load.file_size)
            .ok_or_else(|| "PT_LOAD virtual range overflows u64".to_owned())?;
        if virtual_address < load.virtual_address || virtual_end > load_end {
            continue;
        }
        let delta = virtual_address - load.virtual_address;
        let file_offset = load
            .file_offset
            .checked_add(delta)
            .ok_or_else(|| format!("{label} file offset overflows u64"))?;
        let candidate = checked_range(file_offset, size, file_len, label)?;
        if let Some(previous) = &mapped {
            if previous != &candidate {
                return Err(format!(
                    "{label} maps ambiguously through overlapping PT_LOAD segments"
                ));
            }
        } else {
            mapped = Some(candidate);
        }
    }
    mapped.ok_or_else(|| format!("{label} is not fully backed by a PT_LOAD segment"))
}

fn map_virtual_available(
    loads: &[LoadSegment],
    virtual_address: u64,
    file_len: usize,
    label: &str,
) -> Result<(usize, usize), String> {
    let mut mapped: Option<(usize, usize)> = None;
    for load in loads {
        let load_end = load
            .virtual_address
            .checked_add(load.file_size)
            .ok_or_else(|| "PT_LOAD virtual range overflows u64".to_owned())?;
        if virtual_address < load.virtual_address || virtual_address >= load_end {
            continue;
        }
        let delta = virtual_address - load.virtual_address;
        let file_offset = load
            .file_offset
            .checked_add(delta)
            .ok_or_else(|| format!("{label} file offset overflows u64"))?;
        let available = load.file_size - delta;
        let range = checked_range(file_offset, available, file_len, label)?;
        let candidate = (range.start, range.len());
        if let Some(previous) = mapped {
            if previous != candidate {
                return Err(format!(
                    "{label} maps ambiguously through overlapping PT_LOAD segments"
                ));
            }
        } else {
            mapped = Some(candidate);
        }
    }
    mapped.ok_or_else(|| format!("{label} is not backed by a PT_LOAD segment"))
}

fn set_once(slot: &mut Option<u64>, value: u64, name: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("ELF dynamic table repeats {name}"));
    }
    Ok(())
}

fn required_dynamic_value(value: Option<u64>, name: &str) -> Result<u64, String> {
    value.ok_or_else(|| format!("ELF dynamic table is missing {name}"))
}

fn nul_terminated_bytes<'a>(
    table: &'a [u8],
    offset: usize,
    label: &str,
) -> Result<&'a [u8], String> {
    if offset >= table.len() {
        return Err(format!(
            "{label} offset {offset} exceeds string table length {}",
            table.len()
        ));
    }
    let tail = &table[offset..];
    let length = tail
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(|| format!("{label} is not NUL terminated inside DT_STRSZ"))?;
    Ok(&tail[..length])
}

fn parse_sysv_hash(
    bytes: &[u8],
    loads: &[LoadSegment],
    address: u64,
) -> Result<(usize, NamedRange), String> {
    let prefix = map_virtual_range(loads, address, 8, bytes.len(), "DT_HASH header")?;
    let bucket_count = read_u32(bytes, prefix.start, "DT_HASH nbucket")? as usize;
    let symbol_count = read_u32(bytes, prefix.start + 4, "DT_HASH nchain")? as usize;
    if bucket_count == 0 || bucket_count > MAX_HASH_BUCKETS {
        return Err(format!(
            "DT_HASH bucket count {bucket_count} is outside 1..={MAX_HASH_BUCKETS}"
        ));
    }
    if symbol_count == 0 || symbol_count > MAX_DYNAMIC_SYMBOLS {
        return Err(format!(
            "DT_HASH symbol count {symbol_count} is outside 1..={MAX_DYNAMIC_SYMBOLS}"
        ));
    }
    let word_count = bucket_count
        .checked_add(symbol_count)
        .and_then(|count| count.checked_add(2))
        .ok_or_else(|| "DT_HASH word count overflows usize".to_owned())?;
    let size = word_count
        .checked_mul(4)
        .ok_or_else(|| "DT_HASH byte count overflows usize".to_owned())?;
    let range = map_virtual_range(loads, address, size as u64, bytes.len(), "DT_HASH")?;
    for index in 0..bucket_count + symbol_count {
        let value = read_u32(bytes, range.start + 8 + index * 4, "DT_HASH index")? as usize;
        if value >= symbol_count && value != 0 {
            return Err(format!(
                "DT_HASH index {value} exceeds dynamic symbol count {symbol_count}"
            ));
        }
    }
    Ok((
        symbol_count,
        NamedRange {
            name: "DT_HASH",
            range,
        },
    ))
}

fn parse_gnu_hash(
    bytes: &[u8],
    loads: &[LoadSegment],
    address: u64,
) -> Result<(usize, NamedRange), String> {
    let prefix = map_virtual_range(loads, address, 16, bytes.len(), "DT_GNU_HASH header")?;
    let bucket_count = read_u32(bytes, prefix.start, "DT_GNU_HASH nbuckets")? as usize;
    let symbol_offset = read_u32(bytes, prefix.start + 4, "DT_GNU_HASH symoffset")? as usize;
    let bloom_count = read_u32(bytes, prefix.start + 8, "DT_GNU_HASH bloom_size")? as usize;
    let bloom_shift = read_u32(bytes, prefix.start + 12, "DT_GNU_HASH bloom_shift")?;
    if bucket_count == 0 || bucket_count > MAX_HASH_BUCKETS {
        return Err(format!(
            "DT_GNU_HASH bucket count {bucket_count} is outside 1..={MAX_HASH_BUCKETS}"
        ));
    }
    if symbol_offset == 0 || symbol_offset > MAX_DYNAMIC_SYMBOLS {
        return Err(format!(
            "DT_GNU_HASH symbol offset {symbol_offset} is outside 1..={MAX_DYNAMIC_SYMBOLS}"
        ));
    }
    if bloom_count == 0 || bloom_count > MAX_HASH_BUCKETS || !bloom_count.is_power_of_two() {
        return Err(format!(
            "DT_GNU_HASH bloom count {bloom_count} is not a bounded power of two"
        ));
    }
    if bloom_shift >= 64 {
        return Err(format!(
            "DT_GNU_HASH bloom shift {bloom_shift} is invalid for ELF64"
        ));
    }

    let bloom_bytes = bloom_count
        .checked_mul(8)
        .ok_or_else(|| "DT_GNU_HASH bloom byte count overflows usize".to_owned())?;
    let bucket_bytes = bucket_count
        .checked_mul(4)
        .ok_or_else(|| "DT_GNU_HASH bucket byte count overflows usize".to_owned())?;
    let chain_relative = 16usize
        .checked_add(bloom_bytes)
        .and_then(|offset| offset.checked_add(bucket_bytes))
        .ok_or_else(|| "DT_GNU_HASH prefix byte count overflows usize".to_owned())?;
    let (hash_start, hash_available) =
        map_virtual_available(loads, address, bytes.len(), "DT_GNU_HASH")?;
    if chain_relative > hash_available {
        return Err("DT_GNU_HASH prefix exceeds its PT_LOAD segment".to_owned());
    }
    let buckets_start = hash_start + 16 + bloom_bytes;
    let mut highest_symbol: Option<usize> = None;
    let mut highest_chain_end = chain_relative;
    for bucket_index in 0..bucket_count {
        let symbol = read_u32(
            bytes,
            buckets_start + bucket_index * 4,
            "DT_GNU_HASH bucket",
        )? as usize;
        if symbol == 0 {
            continue;
        }
        if symbol < symbol_offset || symbol >= MAX_DYNAMIC_SYMBOLS {
            return Err(format!(
                "DT_GNU_HASH bucket symbol {symbol} is outside {symbol_offset}..{MAX_DYNAMIC_SYMBOLS}"
            ));
        }
        let mut current = symbol;
        loop {
            if current >= MAX_DYNAMIC_SYMBOLS {
                return Err("DT_GNU_HASH chain exceeds the symbol bound".to_owned());
            }
            let chain_index = current - symbol_offset;
            let relative = chain_relative
                .checked_add(
                    chain_index
                        .checked_mul(4)
                        .ok_or_else(|| "DT_GNU_HASH chain offset overflows usize".to_owned())?,
                )
                .ok_or_else(|| "DT_GNU_HASH chain offset overflows usize".to_owned())?;
            let end = relative
                .checked_add(4)
                .ok_or_else(|| "DT_GNU_HASH chain end overflows usize".to_owned())?;
            if end > hash_available {
                return Err("DT_GNU_HASH chain exceeds its PT_LOAD segment".to_owned());
            }
            let value = read_u32(bytes, hash_start + relative, "DT_GNU_HASH chain")?;
            highest_symbol = Some(highest_symbol.map_or(current, |old| old.max(current)));
            highest_chain_end = highest_chain_end.max(end);
            if value & 1 != 0 {
                break;
            }
            current = current
                .checked_add(1)
                .ok_or_else(|| "DT_GNU_HASH symbol index overflows usize".to_owned())?;
        }
    }
    let symbol_count = highest_symbol
        .map(|index| index + 1)
        .unwrap_or(symbol_offset);
    if symbol_count == 0 || symbol_count > MAX_DYNAMIC_SYMBOLS {
        return Err(format!(
            "DT_GNU_HASH derived symbol count {symbol_count} is outside 1..={MAX_DYNAMIC_SYMBOLS}"
        ));
    }
    Ok((
        symbol_count,
        NamedRange {
            name: "DT_GNU_HASH",
            range: hash_start..hash_start + highest_chain_end,
        },
    ))
}

pub(crate) fn validate_elf64_shared_object(
    bytes: &[u8],
    expected_machine: LinuxMachine,
) -> Result<(), String> {
    if bytes.len() as u64 > MAX_ELF_FILE_SIZE {
        return Err(format!(
            "ELF file size {} exceeds the {} byte validation bound",
            bytes.len(),
            MAX_ELF_FILE_SIZE
        ));
    }
    let header = parse_elf64_header(bytes, expected_machine)?;
    let program_range = table_range(
        header.program_offset,
        header.program_count,
        ELF64_PROGRAM_HEADER_SIZE,
        bytes.len(),
        "ELF program-header table",
    )?;
    let section_range = if header.section_offset == 0 {
        None
    } else {
        Some(table_range(
            header.section_offset,
            header.section_count,
            ELF64_SECTION_HEADER_SIZE,
            bytes.len(),
            "ELF section-header table",
        )?)
    };

    let mut structural_ranges = vec![
        NamedRange {
            name: "ELF header",
            range: 0..ELF64_HEADER_SIZE,
        },
        NamedRange {
            name: "ELF program-header table",
            range: program_range.clone(),
        },
    ];
    if let Some(range) = &section_range {
        structural_ranges.push(NamedRange {
            name: "ELF section-header table",
            range: range.clone(),
        });
    }
    reject_overlapping_ranges(&structural_ranges)?;

    let mut loads = Vec::new();
    let mut dynamic_segment: Option<DynamicSegment> = None;
    for index in 0..header.program_count {
        let offset = program_range.start + index * ELF64_PROGRAM_HEADER_SIZE;
        let segment_type = read_u32(bytes, offset, "program-header p_type")?;
        let file_offset = read_u64(bytes, offset + 8, "program-header p_offset")?;
        let virtual_address = read_u64(bytes, offset + 16, "program-header p_vaddr")?;
        let file_size = read_u64(bytes, offset + 32, "program-header p_filesz")?;
        let memory_size = read_u64(bytes, offset + 40, "program-header p_memsz")?;
        let alignment = read_u64(bytes, offset + 48, "program-header p_align")?;
        if file_size != 0 {
            checked_range(file_offset, file_size, bytes.len(), "program segment")?;
        }
        if alignment > 1 && !alignment.is_power_of_two() {
            return Err(format!(
                "program header {index} has non-power-of-two alignment {alignment}"
            ));
        }
        if alignment > 1 && file_offset % alignment != virtual_address % alignment {
            return Err(format!(
                "program header {index} has incongruent file/virtual alignment"
            ));
        }
        if segment_type == 1 {
            if file_size > memory_size {
                return Err(format!(
                    "PT_LOAD {index} has p_filesz {file_size} greater than p_memsz {memory_size}"
                ));
            }
            loads.push(LoadSegment {
                file_offset,
                virtual_address,
                file_size,
            });
        } else if segment_type == 2 {
            if dynamic_segment.is_some() {
                return Err("ELF contains more than one PT_DYNAMIC segment".to_owned());
            }
            if file_size == 0 || file_size > memory_size {
                return Err("PT_DYNAMIC has an invalid file/memory size".to_owned());
            }
            if file_size % ELF64_DYNAMIC_ENTRY_SIZE as u64 != 0 {
                return Err(format!(
                    "PT_DYNAMIC size {file_size} is not a multiple of {ELF64_DYNAMIC_ENTRY_SIZE}"
                ));
            }
            let entry_count = usize::try_from(file_size / ELF64_DYNAMIC_ENTRY_SIZE as u64)
                .map_err(|_| "PT_DYNAMIC entry count does not fit usize".to_owned())?;
            if entry_count == 0 || entry_count > MAX_DYNAMIC_ENTRIES {
                return Err(format!(
                    "PT_DYNAMIC entry count {entry_count} exceeds {MAX_DYNAMIC_ENTRIES}"
                ));
            }
            dynamic_segment = Some(DynamicSegment {
                file_offset,
                virtual_address,
                file_size,
            });
        }
    }
    if loads.is_empty() {
        return Err("ELF contains no PT_LOAD segment".to_owned());
    }
    let dynamic_segment =
        dynamic_segment.ok_or_else(|| "ELF contains no PT_DYNAMIC segment".to_owned())?;
    let dynamic_range = checked_range(
        dynamic_segment.file_offset,
        dynamic_segment.file_size,
        bytes.len(),
        "PT_DYNAMIC",
    )?;
    let mapped_dynamic = map_virtual_range(
        &loads,
        dynamic_segment.virtual_address,
        dynamic_segment.file_size,
        bytes.len(),
        "PT_DYNAMIC",
    )?;
    if dynamic_range != mapped_dynamic {
        return Err("PT_DYNAMIC file and virtual mappings disagree".to_owned());
    }
    structural_ranges.push(NamedRange {
        name: "PT_DYNAMIC",
        range: dynamic_range.clone(),
    });
    reject_overlapping_ranges(&structural_ranges)?;

    let mut dynamic = DynamicInfo::default();
    let entry_count = dynamic_range.len() / ELF64_DYNAMIC_ENTRY_SIZE;
    let mut terminated = false;
    for index in 0..entry_count {
        let offset = dynamic_range.start + index * ELF64_DYNAMIC_ENTRY_SIZE;
        let tag = read_u64(bytes, offset, "dynamic d_tag")?;
        let value = read_u64(bytes, offset + 8, "dynamic d_val")?;
        if terminated {
            if tag != 0 || value != 0 {
                return Err("PT_DYNAMIC has nonzero entries after DT_NULL".to_owned());
            }
            continue;
        }
        match tag {
            0 => {
                if value != 0 {
                    return Err("DT_NULL has a nonzero value".to_owned());
                }
                terminated = true;
            }
            1 => dynamic.referenced_strings.push(("DT_NEEDED", value)),
            4 => set_once(&mut dynamic.sysv_hash, value, "DT_HASH")?,
            5 => set_once(&mut dynamic.string_table, value, "DT_STRTAB")?,
            6 => set_once(&mut dynamic.symbol_table, value, "DT_SYMTAB")?,
            10 => set_once(&mut dynamic.string_size, value, "DT_STRSZ")?,
            11 => set_once(&mut dynamic.symbol_entry_size, value, "DT_SYMENT")?,
            14 => set_once(&mut dynamic.soname, value, "DT_SONAME")?,
            15 => dynamic.referenced_strings.push(("DT_RPATH", value)),
            29 => dynamic.referenced_strings.push(("DT_RUNPATH", value)),
            0x6fff_fef5 => set_once(&mut dynamic.gnu_hash, value, "DT_GNU_HASH")?,
            0x7fff_fffd => dynamic.referenced_strings.push(("DT_AUXILIARY", value)),
            0x7fff_ffff => dynamic.referenced_strings.push(("DT_FILTER", value)),
            _ => {}
        }
    }
    if !terminated {
        return Err("PT_DYNAMIC is not terminated by DT_NULL".to_owned());
    }

    let string_address = required_dynamic_value(dynamic.string_table, "DT_STRTAB")?;
    let string_size_u64 = required_dynamic_value(dynamic.string_size, "DT_STRSZ")?;
    let string_size =
        usize::try_from(string_size_u64).map_err(|_| "DT_STRSZ does not fit usize".to_owned())?;
    if string_size == 0 || string_size > MAX_DYNAMIC_STRING_TABLE {
        return Err(format!(
            "DT_STRSZ {string_size} is outside 1..={MAX_DYNAMIC_STRING_TABLE}"
        ));
    }
    let string_range = map_virtual_range(
        &loads,
        string_address,
        string_size_u64,
        bytes.len(),
        "DT_STRTAB",
    )?;
    let string_table = &bytes[string_range.clone()];
    if string_table[0] != 0 {
        return Err("DT_STRTAB does not begin with an empty string".to_owned());
    }
    let soname_offset = usize::try_from(required_dynamic_value(dynamic.soname, "DT_SONAME")?)
        .map_err(|_| "DT_SONAME offset does not fit usize".to_owned())?;
    let soname = nul_terminated_bytes(string_table, soname_offset, "DT_SONAME")?;
    if soname != b"libcurl.so.4" {
        return Err(format!(
            "DT_SONAME is {:?}, not libcurl.so.4",
            String::from_utf8_lossy(soname)
        ));
    }
    for (name, offset) in &dynamic.referenced_strings {
        let offset = usize::try_from(*offset)
            .map_err(|_| format!("{name} string offset does not fit usize"))?;
        nul_terminated_bytes(string_table, offset, name)?;
    }

    let symbol_entry_size = required_dynamic_value(dynamic.symbol_entry_size, "DT_SYMENT")?;
    if symbol_entry_size != ELF64_SYMBOL_SIZE as u64 {
        return Err(format!(
            "DT_SYMENT {symbol_entry_size} is not {ELF64_SYMBOL_SIZE}"
        ));
    }
    let mut table_ranges = vec![NamedRange {
        name: "DT_STRTAB",
        range: string_range,
    }];
    let sysv = dynamic
        .sysv_hash
        .map(|address| parse_sysv_hash(bytes, &loads, address))
        .transpose()?;
    let gnu = dynamic
        .gnu_hash
        .map(|address| parse_gnu_hash(bytes, &loads, address))
        .transpose()?;
    let symbol_count = match (&sysv, &gnu) {
        (Some((sysv_count, _)), Some((gnu_count, _))) => {
            if gnu_count > sysv_count {
                return Err(format!(
                    "DT_GNU_HASH derives {gnu_count} symbols but DT_HASH declares only {sysv_count}"
                ));
            }
            *sysv_count
        }
        (Some((count, _)), None) | (None, Some((count, _))) => *count,
        (None, None) => {
            return Err("ELF has neither DT_HASH nor DT_GNU_HASH to bound DT_SYMTAB".to_owned());
        }
    };
    if let Some((_, range)) = sysv {
        table_ranges.push(range);
    }
    if let Some((_, range)) = gnu {
        table_ranges.push(range);
    }

    let symbol_size = symbol_count
        .checked_mul(ELF64_SYMBOL_SIZE)
        .ok_or_else(|| "DT_SYMTAB byte count overflows usize".to_owned())?;
    let symbol_address = required_dynamic_value(dynamic.symbol_table, "DT_SYMTAB")?;
    let symbol_range = map_virtual_range(
        &loads,
        symbol_address,
        symbol_size as u64,
        bytes.len(),
        "DT_SYMTAB",
    )?;
    table_ranges.push(NamedRange {
        name: "DT_SYMTAB",
        range: symbol_range.clone(),
    });
    structural_ranges.extend(table_ranges);
    reject_overlapping_ranges(&structural_ranges)?;

    let mut found = vec![false; REQUIRED_CURL_SYMBOLS.len()];
    for index in 0..symbol_count {
        let offset = symbol_range.start + index * ELF64_SYMBOL_SIZE;
        let name_offset = read_u32(bytes, offset, "dynamic symbol st_name")? as usize;
        let name = nul_terminated_bytes(string_table, name_offset, "dynamic symbol name")?;
        let info = bytes[offset + 4];
        let other = bytes[offset + 5];
        let section_index = read_u16(bytes, offset + 6, "dynamic symbol st_shndx")?;
        let binding = info >> 4;
        let symbol_type = info & 0x0f;
        let visibility = other & 0x03;
        let exported = matches!(binding, 1 | 2)
            && matches!(symbol_type, 0 | 2 | 10)
            && section_index != 0
            && matches!(visibility, 0 | 3);
        if !exported {
            continue;
        }
        for (required_index, required) in REQUIRED_CURL_SYMBOLS.iter().enumerate() {
            if name == *required {
                found[required_index] = true;
            }
        }
    }
    let missing: Vec<String> = REQUIRED_CURL_SYMBOLS
        .iter()
        .zip(found)
        .filter(|(_, present)| !*present)
        .map(|(name, _)| String::from_utf8_lossy(name).into_owned())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "ELF dynamic symbol table is missing required libcurl exports: {}",
            missing.join(", ")
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn validate_owned_mode(path: &Path, expect_directory: bool) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;

    let metadata = fs::metadata(path)
        .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
    if expect_directory && !metadata.is_dir() {
        return Err(format!("{} is not a directory", path.display()));
    }
    if !expect_directory && !metadata.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    if metadata.uid() != 0 {
        return Err(format!("{} is not owned by root", path.display()));
    }
    if metadata.mode() & 0o022 != 0 {
        return Err(format!(
            "{} is writable by group or other users (mode {:o})",
            path.display(),
            metadata.mode() & 0o7777
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_owned_mode(path: &Path, _expect_directory: bool) -> Result<(), String> {
    Err(format!(
        "cannot validate Unix ownership and permissions for {} on this build host",
        path.display()
    ))
}

fn validated_standard_dirs(machine: LinuxMachine) -> Vec<PathBuf> {
    standard_library_dirs(machine)
        .into_iter()
        .filter_map(|directory| {
            let canonical = fs::canonicalize(directory).ok()?;
            validate_owned_mode(&canonical, true).ok()?;
            Some(canonical)
        })
        .collect()
}

fn linux_library_candidates(machine: LinuxMachine) -> Vec<PathBuf> {
    standard_library_dirs(machine)
        .into_iter()
        .flat_map(|directory| [directory.join("libcurl.so"), directory.join("libcurl.so.4")])
        .collect()
}

fn validate_linux_library(
    candidate: &Path,
    allowed_dirs: &[PathBuf],
    machine: LinuxMachine,
) -> Result<PathBuf, String> {
    let canonical = fs::canonicalize(candidate)
        .map_err(|error| format!("cannot canonicalize {}: {error}", candidate.display()))?;
    let parent = canonical
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", canonical.display()))?;
    if !allowed_dirs.iter().any(|allowed| allowed == parent) {
        return Err(format!(
            "{} resolves outside the bounded standard library directories",
            candidate.display()
        ));
    }
    validate_owned_mode(parent, true)?;
    validate_owned_mode(&canonical, false)?;

    let mut file = File::open(&canonical)
        .map_err(|error| format!("cannot open {}: {error}", canonical.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let metadata = file
            .metadata()
            .map_err(|error| format!("cannot inspect open {}: {error}", canonical.display()))?;
        if !metadata.is_file() || metadata.uid() != 0 || metadata.mode() & 0o022 != 0 {
            return Err(format!(
                "open {} is not a root-owned, non-group/world-writable regular file",
                canonical.display()
            ));
        }
        if metadata.len() < ELF64_HEADER_SIZE as u64 || metadata.len() > MAX_ELF_FILE_SIZE {
            return Err(format!(
                "{} has size {}, outside {}..={MAX_ELF_FILE_SIZE}",
                canonical.display(),
                metadata.len(),
                ELF64_HEADER_SIZE
            ));
        }
    }
    let file_size = file
        .metadata()
        .map_err(|error| format!("cannot inspect open {}: {error}", canonical.display()))?
        .len();
    if file_size < ELF64_HEADER_SIZE as u64 || file_size > MAX_ELF_FILE_SIZE {
        return Err(format!(
            "{} has size {file_size}, outside {}..={MAX_ELF_FILE_SIZE}",
            canonical.display(),
            ELF64_HEADER_SIZE
        ));
    }
    let file_size = usize::try_from(file_size)
        .map_err(|_| format!("{} size does not fit this build host", canonical.display()))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(file_size).map_err(|error| {
        format!(
            "cannot reserve {file_size} bytes to validate {}: {error}",
            canonical.display()
        )
    })?;
    bytes.resize(file_size, 0);
    file.read_exact(&mut bytes).map_err(|error| {
        format!(
            "cannot read ELF contents from {}: {error}",
            canonical.display()
        )
    })?;
    validate_elf64_shared_object(&bytes, machine)
        .map_err(|error| format!("{} failed ELF validation: {error}", canonical.display()))?;
    Ok(canonical)
}

pub(crate) fn find_validated_linux_lib(
    machine: LinuxMachine,
) -> Result<(PathBuf, PathBuf), String> {
    let allowed_dirs = validated_standard_dirs(machine);
    if allowed_dirs.is_empty() {
        return Err("no trusted standard Linux library directories are available".to_owned());
    }

    let mut rejected = Vec::new();
    for candidate in linux_library_candidates(machine) {
        if !candidate.exists() {
            continue;
        }
        match validate_linux_library(&candidate, &allowed_dirs, machine) {
            Ok(library) => return Ok((library, candidate)),
            Err(error) => rejected.push(error),
        }
    }

    let detail = if rejected.is_empty() {
        "no libcurl.so or libcurl.so.4 candidate exists".to_owned()
    } else {
        rejected.join("; ")
    };
    Err(format!(
        "no validated system libcurl was found in the bounded standard paths: {detail}. \
         Install the distribution's libcurl runtime/development package"
    ))
}

#[cfg(unix)]
pub(crate) fn install_linker_name(out_dir: &Path, library: &Path) -> Result<PathBuf, String> {
    use std::os::unix::fs::{symlink, DirBuilderExt, PermissionsExt};

    if !library.is_absolute() {
        return Err(format!(
            "validated libcurl path {} is not absolute",
            library.display()
        ));
    }
    let canonical_library = fs::canonicalize(library)
        .map_err(|error| format!("cannot recanonicalize {}: {error}", library.display()))?;
    if canonical_library != library {
        return Err(format!(
            "validated libcurl path {} is no longer canonical (now {})",
            library.display(),
            canonical_library.display()
        ));
    }

    let link_dir = out_dir.join("deltafin-system-libcurl-link-v1");
    match fs::symlink_metadata(&link_dir) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(format!(
                "refusing symlinked linker directory {}",
                link_dir.display()
            ));
        }
        Ok(metadata) if !metadata.is_dir() => {
            return Err(format!(
                "linker directory path {} is not a directory",
                link_dir.display()
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            builder.mode(0o700);
            builder.create(&link_dir).map_err(|create_error| {
                format!("cannot create {}: {create_error}", link_dir.display())
            })?;
        }
        Err(error) => {
            return Err(format!("cannot inspect {}: {error}", link_dir.display()));
        }
    }
    fs::set_permissions(&link_dir, fs::Permissions::from_mode(0o700)).map_err(|error| {
        format!(
            "cannot make linker directory {} private: {error}",
            link_dir.display()
        )
    })?;

    let linker_name = link_dir.join("libcurl.so");
    let needs_install = match fs::symlink_metadata(&linker_name) {
        Ok(metadata) if !metadata.file_type().is_symlink() => {
            return Err(format!(
                "refusing non-symlink linker name {}",
                linker_name.display()
            ));
        }
        Ok(_) => {
            let existing = fs::read_link(&linker_name)
                .map_err(|error| format!("cannot read {}: {error}", linker_name.display()))?;
            existing != library
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
        Err(error) => {
            return Err(format!("cannot inspect {}: {error}", linker_name.display()));
        }
    };
    if needs_install {
        let temporary = link_dir.join(".libcurl.so.deltafin-new");
        match fs::symlink_metadata(&temporary) {
            Ok(metadata) if !metadata.file_type().is_symlink() => {
                return Err(format!(
                    "refusing non-symlink temporary linker name {}",
                    temporary.display()
                ));
            }
            Ok(_) => fs::remove_file(&temporary)
                .map_err(|error| format!("cannot remove stale {}: {error}", temporary.display()))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!("cannot inspect {}: {error}", temporary.display()));
            }
        }
        symlink(library, &temporary)
            .map_err(|error| format!("cannot create {}: {error}", temporary.display()))?;
        if let Err(error) = fs::rename(&temporary, &linker_name) {
            let _ = fs::remove_file(&temporary);
            return Err(format!(
                "cannot atomically install {}: {error}",
                linker_name.display()
            ));
        }
        let metadata = fs::symlink_metadata(&linker_name)
            .map_err(|error| format!("cannot verify {}: {error}", linker_name.display()))?;
        if !metadata.file_type().is_symlink() {
            return Err(format!(
                "atomically installed linker name {} is not a symlink",
                linker_name.display()
            ));
        }
        let installed = fs::read_link(&linker_name)
            .map_err(|error| format!("cannot read {}: {error}", linker_name.display()))?;
        if installed != library {
            return Err(format!(
                "atomically installed {} points to {}, not {}",
                linker_name.display(),
                installed.display(),
                library.display()
            ));
        }
    }
    Ok(link_dir)
}

#[cfg(not(unix))]
pub(crate) fn install_linker_name(_out_dir: &Path, _library: &Path) -> Result<PathBuf, String> {
    Err("a Linux libcurl linker name requires a Unix build host".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_SIZE: usize = 0x2000;
    const FIXTURE_BASE: u64 = 0x400000;
    const PROGRAM_OFFSET: usize = ELF64_HEADER_SIZE;
    const DYNAMIC_OFFSET: usize = 0x200;
    const HASH_OFFSET: usize = 0x300;
    const SYMBOL_OFFSET: usize = 0x500;
    const STRING_OFFSET: usize = 0x900;

    #[derive(Clone, Copy)]
    enum HashFixture {
        Sysv,
        Gnu,
    }

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn elf_header(machine: u16) -> [u8; 64] {
        let mut header = [0_u8; 64];
        header[0..4].copy_from_slice(b"\x7fELF");
        header[4] = 2;
        header[5] = 1;
        header[6] = 1;
        put_u16(&mut header, 16, 3);
        put_u16(&mut header, 18, machine);
        put_u32(&mut header, 20, 1);
        put_u64(&mut header, 32, ELF64_HEADER_SIZE as u64);
        put_u16(&mut header, 52, ELF64_HEADER_SIZE as u16);
        put_u16(&mut header, 54, ELF64_PROGRAM_HEADER_SIZE as u16);
        put_u16(&mut header, 56, 1);
        header
    }

    #[allow(clippy::too_many_arguments)]
    fn put_program_header(
        bytes: &mut [u8],
        index: usize,
        segment_type: u32,
        flags: u32,
        file_offset: u64,
        virtual_address: u64,
        file_size: u64,
        memory_size: u64,
        alignment: u64,
    ) {
        let offset = PROGRAM_OFFSET + index * ELF64_PROGRAM_HEADER_SIZE;
        put_u32(bytes, offset, segment_type);
        put_u32(bytes, offset + 4, flags);
        put_u64(bytes, offset + 8, file_offset);
        put_u64(bytes, offset + 16, virtual_address);
        put_u64(bytes, offset + 24, virtual_address);
        put_u64(bytes, offset + 32, file_size);
        put_u64(bytes, offset + 40, memory_size);
        put_u64(bytes, offset + 48, alignment);
    }

    fn put_dynamic(bytes: &mut [u8], index: usize, tag: u64, value: u64) {
        let offset = DYNAMIC_OFFSET + index * ELF64_DYNAMIC_ENTRY_SIZE;
        put_u64(bytes, offset, tag);
        put_u64(bytes, offset + 8, value);
    }

    fn shared_object_fixture(machine: LinuxMachine, hash_kind: HashFixture) -> Vec<u8> {
        let mut bytes = vec![0_u8; FIXTURE_SIZE];
        bytes[..ELF64_HEADER_SIZE].copy_from_slice(&elf_header(machine.elf_machine()));
        put_u16(&mut bytes, 56, 2);

        let dynamic_entries = 7_u64;
        let dynamic_size = dynamic_entries * ELF64_DYNAMIC_ENTRY_SIZE as u64;
        put_program_header(
            &mut bytes,
            0,
            1,
            5,
            0,
            FIXTURE_BASE,
            FIXTURE_SIZE as u64,
            FIXTURE_SIZE as u64,
            0x1000,
        );
        put_program_header(
            &mut bytes,
            1,
            2,
            6,
            DYNAMIC_OFFSET as u64,
            FIXTURE_BASE + DYNAMIC_OFFSET as u64,
            dynamic_size,
            dynamic_size,
            8,
        );

        let mut strings = vec![0_u8];
        let soname_offset = strings.len();
        strings.extend_from_slice(b"libcurl.so.4\0");
        let mut name_offsets = Vec::new();
        for name in REQUIRED_CURL_SYMBOLS {
            name_offsets.push(strings.len());
            strings.extend_from_slice(name);
            strings.push(0);
        }
        bytes[STRING_OFFSET..STRING_OFFSET + strings.len()].copy_from_slice(&strings);

        for (index, name_offset) in name_offsets.iter().enumerate() {
            let offset = SYMBOL_OFFSET + (index + 1) * ELF64_SYMBOL_SIZE;
            put_u32(&mut bytes, offset, *name_offset as u32);
            bytes[offset + 4] = 0x12;
            bytes[offset + 5] = 0;
            put_u16(&mut bytes, offset + 6, 1);
            put_u64(&mut bytes, offset + 8, FIXTURE_BASE + 0x1000 + index as u64);
            put_u64(&mut bytes, offset + 16, 1);
        }
        let symbol_count = REQUIRED_CURL_SYMBOLS.len() + 1;

        match hash_kind {
            HashFixture::Sysv => {
                put_u32(&mut bytes, HASH_OFFSET, 1);
                put_u32(&mut bytes, HASH_OFFSET + 4, symbol_count as u32);
                put_u32(&mut bytes, HASH_OFFSET + 8, 1);
            }
            HashFixture::Gnu => {
                put_u32(&mut bytes, HASH_OFFSET, 1);
                put_u32(&mut bytes, HASH_OFFSET + 4, 1);
                put_u32(&mut bytes, HASH_OFFSET + 8, 1);
                put_u32(&mut bytes, HASH_OFFSET + 12, 6);
                put_u64(&mut bytes, HASH_OFFSET + 16, 0);
                put_u32(&mut bytes, HASH_OFFSET + 24, 1);
                for index in 0..REQUIRED_CURL_SYMBOLS.len() {
                    let value = if index + 1 == REQUIRED_CURL_SYMBOLS.len() {
                        1
                    } else {
                        0
                    };
                    put_u32(&mut bytes, HASH_OFFSET + 28 + index * 4, value);
                }
            }
        }

        put_dynamic(&mut bytes, 0, 5, FIXTURE_BASE + STRING_OFFSET as u64);
        put_dynamic(&mut bytes, 1, 6, FIXTURE_BASE + SYMBOL_OFFSET as u64);
        put_dynamic(&mut bytes, 2, 10, strings.len() as u64);
        put_dynamic(&mut bytes, 3, 11, ELF64_SYMBOL_SIZE as u64);
        put_dynamic(&mut bytes, 4, 14, soname_offset as u64);
        put_dynamic(
            &mut bytes,
            5,
            match hash_kind {
                HashFixture::Sysv => 4,
                HashFixture::Gnu => 0x6fff_fef5,
            },
            FIXTURE_BASE + HASH_OFFSET as u64,
        );
        put_dynamic(&mut bytes, 6, 0, 0);
        bytes
    }

    #[test]
    fn target_allowlist_is_exact() {
        assert!(ensure_native_build("aarch64-apple-darwin", "aarch64-apple-darwin").is_ok());
        assert!(ensure_native_build("aarch64-apple-darwin", "aarch64-unknown-linux-gnu").is_err());
        assert_eq!(
            parse_supported_target("aarch64-apple-darwin"),
            Ok(Target::MacArm64)
        );
        assert_eq!(
            parse_supported_target("x86_64-unknown-linux-gnu"),
            Ok(Target::Linux(LinuxMachine::X86_64))
        );
        assert_eq!(
            parse_supported_target("aarch64-unknown-linux-gnu"),
            Ok(Target::Linux(LinuxMachine::Aarch64))
        );
        assert!(parse_supported_target("x86_64-unknown-linux-musl").is_err());
        assert!(parse_supported_target("x86_64-pc-windows-msvc").is_err());
    }

    #[test]
    fn valid_elf_headers_match_only_the_requested_machine() {
        let x86 = elf_header(62);
        let arm = elf_header(183);
        assert!(validate_elf64_header(&x86, LinuxMachine::X86_64).is_ok());
        assert!(validate_elf64_header(&arm, LinuxMachine::Aarch64).is_ok());
        assert!(validate_elf64_header(&x86, LinuxMachine::Aarch64).is_err());
        assert!(validate_elf64_header(&arm, LinuxMachine::X86_64).is_err());
    }

    #[test]
    fn elf_validation_rejects_scripts_wrong_class_and_non_shared_objects() {
        assert!(validate_elf64_header(b"GROUP ( libcurl.so.4 )", LinuxMachine::X86_64).is_err());

        let mut wrong_class = elf_header(62);
        wrong_class[4] = 1;
        assert!(validate_elf64_header(&wrong_class, LinuxMachine::X86_64).is_err());

        let mut executable = elf_header(62);
        put_u16(&mut executable, 16, 2);
        assert!(validate_elf64_header(&executable, LinuxMachine::X86_64).is_err());

        let mut old_version = elf_header(62);
        put_u32(&mut old_version, 20, 0);
        assert!(validate_elf64_header(&old_version, LinuxMachine::X86_64).is_err());

        let mut bad_header_size = elf_header(62);
        put_u16(&mut bad_header_size, 52, 63);
        assert!(validate_elf64_header(&bad_header_size, LinuxMachine::X86_64).is_err());
    }

    #[test]
    fn unsupported_feature_set_is_reported_together() {
        let unsupported = [
            "http2",
            "mesalink",
            "ntlm",
            "poll_7_68_0",
            "protocol-ftp",
            "rustls",
            "spnego",
            "static-curl",
            "static-ssl",
            "upkeep_7_62_0",
            "windows-static-ssl",
            "zlib-ng-compat",
        ];
        assert!(ensure_supported_features(
            &unsupported
                .iter()
                .map(|name| (*name, false))
                .collect::<Vec<_>>()
        )
        .is_ok());
        let enabled: Vec<(&str, bool)> = unsupported.iter().map(|name| (*name, true)).collect();
        let error = ensure_supported_features(&enabled)
            .expect_err("capability-changing features must fail closed");
        for feature in unsupported {
            assert!(
                error.contains(feature),
                "missing {:?} from {:?}",
                feature,
                error
            );
        }
    }

    #[test]
    fn complete_sysv_and_gnu_hash_fixtures_validate_for_both_linux_machines() {
        for machine in [LinuxMachine::X86_64, LinuxMachine::Aarch64] {
            for hash in [HashFixture::Sysv, HashFixture::Gnu] {
                validate_elf64_shared_object(&shared_object_fixture(machine, hash), machine)
                    .expect("bounded synthetic libcurl should validate");
            }
        }
    }

    #[test]
    fn full_validation_rejects_missing_or_malformed_dynamic_structures() {
        let mut no_dynamic = shared_object_fixture(LinuxMachine::X86_64, HashFixture::Sysv);
        put_u32(
            &mut no_dynamic,
            PROGRAM_OFFSET + ELF64_PROGRAM_HEADER_SIZE,
            0,
        );
        assert!(validate_elf64_shared_object(&no_dynamic, LinuxMachine::X86_64).is_err());

        let mut no_null = shared_object_fixture(LinuxMachine::X86_64, HashFixture::Sysv);
        put_dynamic(&mut no_null, 6, 1, 0);
        assert!(validate_elf64_shared_object(&no_null, LinuxMachine::X86_64).is_err());

        let mut out_of_file = shared_object_fixture(LinuxMachine::X86_64, HashFixture::Sysv);
        put_dynamic(
            &mut out_of_file,
            0,
            5,
            FIXTURE_BASE + FIXTURE_SIZE as u64 + 1,
        );
        assert!(validate_elf64_shared_object(&out_of_file, LinuxMachine::X86_64).is_err());

        let mut overflowed_program_table =
            shared_object_fixture(LinuxMachine::X86_64, HashFixture::Sysv);
        put_u64(&mut overflowed_program_table, 32, u64::MAX - 8);
        assert!(
            validate_elf64_shared_object(&overflowed_program_table, LinuxMachine::X86_64).is_err()
        );

        let mut bad_needed = shared_object_fixture(LinuxMachine::X86_64, HashFixture::Sysv);
        let dynamic_header = PROGRAM_OFFSET + ELF64_PROGRAM_HEADER_SIZE;
        put_u64(
            &mut bad_needed,
            dynamic_header + 32,
            8 * ELF64_DYNAMIC_ENTRY_SIZE as u64,
        );
        put_u64(
            &mut bad_needed,
            dynamic_header + 40,
            8 * ELF64_DYNAMIC_ENTRY_SIZE as u64,
        );
        put_dynamic(&mut bad_needed, 6, 1, u64::MAX);
        put_dynamic(&mut bad_needed, 7, 0, 0);
        assert!(validate_elf64_shared_object(&bad_needed, LinuxMachine::X86_64).is_err());
    }

    #[test]
    fn full_validation_requires_exact_soname_and_every_export() {
        let mut wrong_soname = shared_object_fixture(LinuxMachine::X86_64, HashFixture::Sysv);
        wrong_soname[STRING_OFFSET + 1] = b'x';
        let error = validate_elf64_shared_object(&wrong_soname, LinuxMachine::X86_64)
            .expect_err("wrong SONAME must fail");
        assert!(error.contains("DT_SONAME"));

        let mut missing_export = shared_object_fixture(LinuxMachine::X86_64, HashFixture::Sysv);
        let name = REQUIRED_CURL_SYMBOLS[REQUIRED_CURL_SYMBOLS.len() - 1];
        let location = missing_export
            .windows(name.len())
            .position(|window| window == name)
            .expect("fixture contains required name");
        missing_export[location] = b'x';
        let error = validate_elf64_shared_object(&missing_export, LinuxMachine::X86_64)
            .expect_err("missing export must fail");
        assert!(error.contains("missing required libcurl exports"));
    }

    #[test]
    fn full_validation_rejects_overlapping_tables_and_unbounded_hashes() {
        let mut overlap = shared_object_fixture(LinuxMachine::X86_64, HashFixture::Sysv);
        put_dynamic(&mut overlap, 1, 6, FIXTURE_BASE + STRING_OFFSET as u64);
        let error = validate_elf64_shared_object(&overlap, LinuxMachine::X86_64)
            .expect_err("overlapping string/symbol tables must fail");
        assert!(error.contains("overlap"));

        let mut too_many_symbols = shared_object_fixture(LinuxMachine::X86_64, HashFixture::Sysv);
        put_u32(
            &mut too_many_symbols,
            HASH_OFFSET + 4,
            MAX_DYNAMIC_SYMBOLS as u32 + 1,
        );
        assert!(validate_elf64_shared_object(&too_many_symbols, LinuxMachine::X86_64).is_err());

        let mut unterminated_gnu = shared_object_fixture(LinuxMachine::X86_64, HashFixture::Gnu);
        for index in 0..REQUIRED_CURL_SYMBOLS.len() {
            put_u32(&mut unterminated_gnu, HASH_OFFSET + 28 + index * 4, 0);
        }
        assert!(validate_elf64_shared_object(&unterminated_gnu, LinuxMachine::X86_64).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn private_linker_name_is_idempotent_and_atomically_retargets() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("deltafin-curl-link-test-{nonce}"));
        fs::create_dir(&root).expect("create isolated test directory");
        let root = fs::canonicalize(&root).expect("canonicalize isolated test directory");
        let library = root.join("libcurl.so.4.known");
        fs::write(&library, b"test").expect("create inert test target");

        let link_dir = install_linker_name(&root, &library).expect("create linker name");
        assert_eq!(
            fs::read_link(link_dir.join("libcurl.so")).expect("read linker name"),
            library
        );
        assert_eq!(
            fs::metadata(&link_dir)
                .expect("link directory metadata")
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            install_linker_name(&root, &library).expect("reuse exact linker name"),
            link_dir
        );

        let other = root.join("other-libcurl.so.4");
        fs::write(&other, b"new").expect("create replacement inert target");
        assert_eq!(
            install_linker_name(&root, &other).expect("atomically retarget linker name"),
            link_dir
        );
        assert_eq!(
            fs::read_link(link_dir.join("libcurl.so")).expect("read retargeted linker name"),
            other
        );
        assert!(!link_dir.join(".libcurl.so.deltafin-new").exists());

        fs::remove_file(link_dir.join("libcurl.so")).expect("remove linker symlink");
        fs::write(link_dir.join("libcurl.so"), b"do not replace")
            .expect("create guarded non-symlink");
        assert!(install_linker_name(&root, &library).is_err());

        fs::remove_file(link_dir.join("libcurl.so")).expect("remove guarded test file");
        let temporary = link_dir.join(".libcurl.so.deltafin-new");
        fs::write(&temporary, b"do not replace").expect("create guarded temporary file");
        assert!(install_linker_name(&root, &library).is_err());
        fs::remove_file(&temporary).expect("remove guarded temporary file");

        let alias = root.join("libcurl.so.4.alias");
        symlink(&library, &alias).expect("create noncanonical library alias");
        assert!(install_linker_name(&root, &alias).is_err());
        fs::remove_file(&alias).expect("remove noncanonical library alias");

        fs::remove_dir(&link_dir).expect("remove private test link directory");
        fs::remove_file(&library).expect("remove inert test target");
        fs::remove_file(&other).expect("remove replacement inert target");
        fs::remove_dir(&root).expect("remove isolated test directory");
    }
}
