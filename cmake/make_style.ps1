param(
  [Parameter(Mandatory = $true)][string]$OutDir,
  [Parameter(Mandatory = $true)][string]$SrcDir
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$styleDefs = @(
  @{ Token = "COLLIDE_CLASS"; Prefix = "collide_"; Out = "collide" },
  @{ Token = "COMMAND_CLASS"; Prefix = ""; Out = "command" },
  @{ Token = "COMPUTE_CLASS"; Prefix = "compute_"; Out = "compute" },
  @{ Token = "DUMP_CLASS"; Prefix = "dump_"; Out = "dump" },
  @{ Token = "FIX_CLASS"; Prefix = "fix_"; Out = "fix" },
  @{ Token = "REACT_CLASS"; Prefix = "react_"; Out = "react" },
  @{ Token = "REGION_CLASS"; Prefix = "region_"; Out = "region" },
  @{ Token = "SURF_COLLIDE_CLASS"; Prefix = "surf_collide_"; Out = "surf_collide" },
  @{ Token = "SURF_REACT_CLASS"; Prefix = "surf_react_"; Out = "surf_react" }
)

foreach ($def in $styleDefs) {
  $files = Get-ChildItem -Path $SrcDir -Filter "$($def.Prefix)*.h" -File |
    Where-Object { Select-String -Path $_.FullName -Pattern $def.Token -SimpleMatch -Quiet } |
    Sort-Object Name

  $outFile = Join-Path $OutDir ("style_{0}.h" -f $def.Out)
  if ($files.Count -eq 0) {
    Set-Content -Path $outFile -Value $null -NoNewline
    continue
  }

  $lines = foreach ($file in $files) {
    '#include "{0}"' -f $file.Name
  }

  $content = ($lines -join [Environment]::NewLine) + [Environment]::NewLine
  Set-Content -Path $outFile -Value $content -NoNewline
}
