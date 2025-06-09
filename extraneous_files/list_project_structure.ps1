# list_project_structure.ps1
# Script to recursively list project structure (folders and .py files)

param (
    [string]$RootPath = ".",  # Default to current directory
    [string]$OutputFile = "project_structure.txt",
    [int]$MaxDepth = 7 # Adjust as needed to prevent excessively deep recursion in large projects
)

function Get-Tree {
    param (
        [string]$Path,
        [string]$Indent = "",
        [int]$CurrentDepth = 0
    )

    if ($CurrentDepth -gt $MaxDepth) {
        return "$Indent`-- [Max Depth Reached]" # Indicate if max depth is hit
    }

    # List directories first, then files
    Get-ChildItem -Path $Path -Directory | ForEach-Object {
        "$Indent`-- $($_.Name)/"
        Get-Tree -Path $_.FullName -Indent "$Indent   |" -CurrentDepth ($CurrentDepth + 1)
    }
    Get-ChildItem -Path $Path -File | Where-Object { $_.Extension -eq ".py" -or $_.Name -eq "__init__.py" -or $_.Name -like "*.yaml" -or $_.Name -like "*.json" -or $_.Name -like "*.txt" -or $_.Name -like "*.md"} | ForEach-Object { # Focus on relevant files
        "$Indent`-- $($_.Name)"
    }
}

# Start the process
$outputContent = "Project Structure for: $(Resolve-Path $RootPath)"
$outputContent += "`n------------------------------------------`n"
$outputContent += (Get-Tree -Path $RootPath)

# Save to output file
try {
    Set-Content -Path $OutputFile -Value $outputContent -Encoding UTF8 -Force
    Write-Host "Project structure saved to '$OutputFile'"
}
catch {
    Write-Error "Error saving project structure: $_"
}

# Optional: Display on console
# $outputContent