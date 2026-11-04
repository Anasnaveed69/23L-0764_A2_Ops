#!/usr/bin/env bash
# Task 7 — create the VisionGuard API VM on Azure for Students, from Azure Cloud Shell (Bash).
#   curl -fsSL https://raw.githubusercontent.com/Anasnaveed69/23L-0764_A2_Ops/main/deploy/azure_cloudshell.sh | bash
# Every step is idempotent: re-running it skips what already exists.
set -euo pipefail

RG=rg-23L-0764-A2-Ops
VM=vm-23L-0764-A2-Ops
SIZE=Standard_B2s                  # 2 vCPU / 4 GB: TensorFlow needs > 1 GB RAM at start-up
IMAGE=Ubuntu2404                   # x86-64: same tensorflow-cpu wheel as CI and the laptop
# Azure for Students only allows some regions; try nearby ones first.
REGIONS="${REGIONS:-uaenorth centralindia southeastasia eastus westeurope}"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

say "subscription"
az account show --query "{name:name, id:id, user:user.name}" -o table

if az vm show -g "$RG" -n "$VM" >/dev/null 2>&1; then
  say "VM $VM already exists"
else
  for LOC in $REGIONS; do
    say "trying region $LOC"
    az group create -n "$RG" -l "$LOC" -o none || continue
    if az vm create -g "$RG" -n "$VM" -l "$LOC" --image "$IMAGE" --size "$SIZE" \
         --admin-username azureuser --generate-ssh-keys --public-ip-sku Standard \
         --nsg-rule SSH --os-disk-size-gb 30 -o table; then
      break
    fi
    az group delete -n "$RG" --yes --no-wait || true
    sleep 20
  done
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
