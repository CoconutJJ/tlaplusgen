#!/usr/bin/env bash
# Run sass2tla.py over every kernel in every example .sass file and record
# pass/fail plus any stderr output. Outputs land in out/ under the repo root.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/src"
OUT="$HERE/out"
TLC_JAR="$HERE/tla2tools_i64.jar"
TLC_TIMEOUT="${TLC_TIMEOUT:-120}"
export PYTHONPATH="$SRC/sass/gpucode-analyzer${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

fail=0
pass=0
skipped=0
tlc_ok=0
tlc_finding=0
tlc_timeout=0
tlc_overflow=0
tlc_error=0

for sass in "$HERE"/examples/regalloc-dataset/regalloc/*.sass; do
  [ -f "$sass" ] || continue
  base="$(basename "$sass" .sass)"
  echo "=== $base ==="

  # Ask sass2tla.py to list kernels. It exits 0 and prints a line per kernel
  # when --kernel is omitted.
  mapfile -t listing < <(cd "$SRC" && python sass2tla.py "$sass" --regs_per_thread 64 2>/dev/null)
  kernels=()
  for line in "${listing[@]}"; do
    case "$line" in
      "Please select a kernel"*|""|"[codegen]"*) continue ;;
      *) kernels+=("$line") ;;
    esac
  done

  if [ ${#kernels[@]} -eq 0 ]; then
    echo "  (no kernels found, skipping)"
    skipped=$((skipped + 1))
    continue
  fi

  idx=0
  for kernel in "${kernels[@]}"; do
    idx=$((idx + 1))
    module="Run_${base//./_}_k${idx}"
    module="${module:0:64}"
    log="$OUT/${module}.log"
    echo "  -> [$idx] $kernel"
    if (cd "$SRC" && python sass2tla.py "$sass" --regs_per_thread 64 \
         --kernel "$kernel" --module "$module") >"$log" 2>&1; then

      if [ -f "$SRC/${module}.tla" ]; then
        
        mv "$SRC/${module}.tla" "$SRC/${module}.cfg" "$OUT/" 2>/dev/null || true
        
        pass=$((pass + 1))
        
        echo "     OK (generated)"

        tlc_log="$OUT/${module}.tlc.log"
        (cd "$OUT" && timeout "$TLC_TIMEOUT" java \
              -XX:+UseParallelGC -jar "$TLC_JAR" \
              -config "${module}.cfg" "${module}.tla") \
          >"$tlc_log" 2>&1
        rc=$?
        case "$rc" in
          0)
            tlc_ok=$((tlc_ok + 1))
            echo "        TLC OK (no violations)"
            ;;
          12|13)
            tlc_finding=$((tlc_finding + 1))
            finding="$(grep -Eo 'Invariant [A-Za-z0-9_]+ is violated|Temporal properties.*violated|Deadlock reached' "$tlc_log" | head -1)"
            echo "        TLC ran: ${finding:-spec violation} (see $tlc_log)"
            ;;
          124)
            tlc_timeout=$((tlc_timeout + 1))
            echo "        TLC TIMEOUT after ${TLC_TIMEOUT}s (see $tlc_log)"
            ;;
          *)
            if grep -q "Overflow when computing" "$tlc_log"; then
              tlc_overflow=$((tlc_overflow + 1))
              echo "        TLC int64 overflow (handler produces > 2^63) (see $tlc_log)"
            else
              tlc_error=$((tlc_error + 1))
              echo "        TLC ERROR rc=$rc (see $tlc_log)"
            fi
            ;;
        esac
      else
        skipped=$((skipped + 1))
        echo "     SKIP (empty slice)"
      fi
    else
      fail=$((fail + 1))
      echo "     FAILED (see $log)"
    fi
  done
done

echo
echo "codegen:  pass=$pass  fail=$fail  skipped=$skipped"
echo "TLC:      ok=$tlc_ok  finding=$tlc_finding  overflow=$tlc_overflow  timeout=$tlc_timeout  error=$tlc_error"
[ "$fail" -eq 0 ] && [ "$tlc_error" -eq 0 ]
