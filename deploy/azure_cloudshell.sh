#!/usr/bin/env bash
# Task 7 — create the VisionGuard API VM on Azure for Students, from Azure Cloud Shell (Bash).
#   curl -fsSL https://raw.githubusercontent.com/Anasnaveed69/23L-0764_A2_Ops/main/deploy/azure_cloudshell.sh | bash
# Every step is idempotent: re-running it skips what already exists.
set -euo pipefail

RG=rg-23L-0764-A2-Ops
VM=vm-23L-0764-A2-Ops
# 2 vCPU / 4-8 GB: TensorFlow needs > 1 GB RAM at start-up. Student regions often lack capacity for
# one size, so equivalent small sizes are tried in order.
SIZES="${SIZES:-Standard_B2s Standard_B2s_v2 Standard_B2as_v2 Standard_B2ls_v2 Standard_D2s_v5 Standard_D2as_v5}"
IMAGE=Ubuntu2404                   # x86-64: same tensorflow-cpu wheel as CI and the laptop
# Azure for Students has an "Allowed resource deployment regions" policy; read it, nearby first.
ALLOWED=$(az policy assignment list --query "[?displayName=='Allowed resource deployment regions'].parameters.listOfAllowedLocations.value[]" -o tsv | tr '\n' ' ')
REGIONS="${REGIONS:-uaenorth centralindia southeastasia malaysiawest austriaeast}"
if [ -n "$ALLOWED" ]; then
  REGIONS=$(for r in $REGIONS; do if [[ " $ALLOWED " == *" $r "* ]]; then echo -n "$r "; fi; done)
fi

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

say "subscription"
az account show --query "{name:name, id:id, user:user.name}" -o table
echo "allowed regions: $ALLOWED"

say "register resource providers (new subscriptions start with them unregistered)"
for p in Microsoft.Compute Microsoft.Network Microsoft.Storage; do
  az provider register -n "$p" --wait -o none && echo "$p: $(az provider show -n $p --query registrationState -o tsv)"
done

if az vm show -g "$RG" -n "$VM" >/dev/null 2>&1; then
  say "VM $VM already exists"
else
  if [ "$(az group exists -n "$RG")" = "true" ]; then
    say "removing leftover resource group from an earlier failed attempt"
    az group delete -n "$RG" --yes
  fi
  created=""
  for LOC in $REGIONS; do
    # only sizes this subscription can actually deploy in this region right now
    AVAIL=$(az vm list-skus -l "$LOC" --resource-type virtualMachines --all false \
            --query "[?length(restrictions)==\`0\`].name" -o tsv 2>/dev/null | tr '\n' ' ')
    for SIZE in $SIZES; do
      [[ " $AVAIL " == *" $SIZE "* ]] || continue
      say "trying $SIZE in $LOC"
      az group create -n "$RG" -l "$LOC" -o none || continue
      if az vm create -g "$RG" -n "$VM" -l "$LOC" --image "$IMAGE" --size "$SIZE" \
           --admin-username azureuser --generate-ssh-keys --public-ip-sku Standard \
           --nsg-rule SSH --os-disk-size-gb 30 -o table 2>&1 | grep -v '^  File \|^    '; then
        if az vm show -g "$RG" -n "$VM" >/dev/null 2>&1; then created=1; break 2; fi
      fi
      az group delete -n "$RG" --yes || true
    done
  done
  [ -n "$created" ] || { echo "no region/size combination worked"; exit 1; }
fi

say "cloud firewall (Network Security Group): allow 80 and 443"
az vm open-port -g "$RG" -n "$VM" --port 80  --priority 1010 -o none || true
az vm open-port -g "$RG" -n "$VM" --port 443 --priority 1020 -o none || true
NSG=$(az network nsg list -g "$RG" --query "[0].name" -o tsv)
az network nsg rule list -g "$RG" --nsg-name "$NSG" \
  --query "[].{rule:name, port:destinationPortRange, access:access, priority:priority}" -o table

IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
DOMAIN="${IP//./-}.sslip.io"
say "public IP $IP  ->  https://$DOMAIN"
echo "ssh azureuser@$IP"
