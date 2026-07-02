# GitHub publishing

The toolkit is already a clean git repository. Current commit:

```text
a64679c Add config-driven gen screening toolkit
```

## Publish from Windows PowerShell

```powershell
cd E:\codex_project\mid_term\tools\gen_screening
.\scripts\publish_to_github.ps1 -RemoteUrl git@github.com:OWNER/REPO.git
```

For HTTPS token authentication:

```powershell
.\scripts\publish_to_github.ps1 -RemoteUrl https://github.com/OWNER/REPO.git
```

## Publish from the AutoDL server

```bash
cd /root/autodl-tmp/road_damage_exp/tools/gen_screening
bash scripts/publish_to_github.sh git@github.com:OWNER/REPO.git
```

## Current blocker

The current AutoDL server does not have GitHub SSH access:

```text
git@github.com: Permission denied (publickey).
```

The current Codex GitHub connector also lists zero accessible repositories. A real `OWNER/REPO` plus working GitHub credentials are required before the final push can be verified.
