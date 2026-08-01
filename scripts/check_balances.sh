#!/usr/bin/env bash
# check_balances.sh — query every provider for its remaining balance.
# Prints a table; never echoes a key. Refreshes the numbers in Recharge.md.
#
#   bash scripts/check_balances.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load whichever .env is present (Phase-2 one is the superset).
for cand in "$REPO_ROOT/../Code_Phase_2/.env" "$REPO_ROOT/Code_Phase_2/.env" "$REPO_ROOT/.env"; do
    if [ -f "$cand" ]; then set -a; . "$cand" >/dev/null 2>&1; set +a; fi
done

jqf() { python3 -c "import sys,json;d=json.load(sys.stdin);print(eval(sys.argv[1],{'d':d}))" "$1" 2>/dev/null || echo "?"; }

printf '%-22s %s\n' "PROVIDER" "BALANCE"
printf '%-22s %s\n' "----------------------" "-------"

if [ -n "${DEEPSEEK_API_KEY_1:-}" ]; then
    b=$(curl -s -m 30 -H "Authorization: Bearer $DEEPSEEK_API_KEY_1" \
        https://api.deepseek.com/user/balance |
        jqf "d['balance_infos'][0]['total_balance']+' '+d['balance_infos'][0]['currency']")
    printf '%-22s %s\n' "DeepSeek" "$b"
fi

for n in 1 2; do
    var="OPENROUTER_API_KEY_$n"; key="${!var:-}"
    [ -z "$key" ] && continue
    c=$(curl -s -m 30 -H "Authorization: Bearer $key" https://openrouter.ai/api/v1/credits |
        jqf "'%.2f'%(d['data']['total_credits']-d['data']['total_usage'])")
    l=$(curl -s -m 30 -H "Authorization: Bearer $key" https://openrouter.ai/api/v1/key |
        jqf "d['data'].get('limit_remaining')")
    printf '%-22s $%s credit (key spend-limit remaining: %s)\n' "OpenRouter #$n" "$c" "$l"
done

# Vast.ai key lives in the Phase-1 .env as PHD_VAST_AI_KEY
if [ -n "${PHD_VAST_AI_KEY:-}" ]; then
    v=$(curl -s -m 30 -H "Authorization: Bearer $PHD_VAST_AI_KEY" \
        https://console.vast.ai/api/v0/users/current/ |
        jqf "'%.2f'%(d.get('credit') or 0)")
    printf '%-22s $%s\n' "Vast.ai" "$v"
fi

# These providers expose no balance endpoint — report reachability only.
probe() {  # name url auth_header
    [ -z "${3:-}" ] && return
    code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $3" "$2")
    printf '%-22s key %s (no balance endpoint — check dashboard)\n' "$1" \
        "$([ "$code" = 200 ] && echo valid || echo "FAILED http=$code")"
}
probe "LinkAPI"  "https://api.linkapi.ai/v1/models"     "${AZUREOPENAI_LINKAPI_KEY:-}"
probe "nano-gpt" "https://nano-gpt.com/api/v1/models"   "${Nano_GPT_API_KEY:-}"
probe "Mistral"  "https://api.mistral.ai/v1/models"     "${MISTRAL_API_KEY_1:-${MISTRAL_API_KEY:-}}"
