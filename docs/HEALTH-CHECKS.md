# Health checks

These standalone auditors are read-only and network-free; run them after an install or upgrade, or when diagnosing a problem, to verify the runtime and each installed component in place:

```bash
./target/release/deltafin setup-dspark --check
./target/release/deltafin setup-qwen --check --model-root .
./target/release/deltafin setup-k3 --check --model-root .
./target/release/deltafin doctor --runtime-only
```

`doctor --runtime-only` validates the host, compiled executable, provider ABI and native canaries without opening or scanning model data. Omit that flag for the normal complete installation audit. Standalone K3 and Qwen setup commands accept `--model-root`; without it they retain the `DELTAFIN_ROOT`/current-folder fallback.
