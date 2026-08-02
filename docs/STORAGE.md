# Native storage preparation

The default resident spine is the measured **row-int8** representation. `deltafin setup` prepares it automatically at the end of installation by converting the downloaded BF16 checkpoint, so a fresh install runs the default with no separate step. The conversion is resumable, authenticated, and recognizes an already-complete output. Existing installations can produce it explicitly:

```bash
./target/release/deltafin convert-spine-int8
```

Int8 changes resident weights and is therefore still labeled **quantized and non-weight-exact** throughout the CLI. The released MXFP4 routed experts are untouched in every configuration — K3's native representation, all 16 always routed.

The **original BF16 spine remains on disk** as the conversion source and verification authority, and stays selectable explicitly:

```bash
./target/release/deltafin run --spine bf16 --prompt "The capital of France is" --max-new 17
```

While accuracy validation of the quantized default continues, selection never falls back silently: if the int8 spine is missing, `--spine auto` fails with instructions instead of substituting a different representation.

Either spine can be packed into authenticated DFSP files to reduce file discovery and make layer reads contiguous:

```bash
./target/release/deltafin pack-spine --spine int8
./target/release/deltafin pack-spine --spine int8 --verify-only
./target/release/deltafin pack-spine --spine bf16
```

Full local installs may losslessly compact only the MXFP4 scale streams:

```bash
./target/release/deltafin convert-experts-scale4
```

The resumable conversion adds about **40.25 GiB** of sidecars and keeps the raw experts. Activation is atomic only after the complete 82,432-expert corpus validates. Packed expert values are unchanged and scales reconstruct exactly.

Approximate disk footprint on a full install: 1.7 TB raw experts, 107 GB BF16 spine source, 53 GB int8 resident spine, plus optional scale4 sidecars and DFSP packs.
