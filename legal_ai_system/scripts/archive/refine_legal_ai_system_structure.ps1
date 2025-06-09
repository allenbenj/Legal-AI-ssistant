# refine_legal_ai_system_structure.ps1
# Targets specific files for moving them into their correct subdirectories.

param (
    [string]$ProjectRoot = (Get-Location).Path 
)

$LegalSystemBaseDirName = "legal_ai_system"
$LegalSystemFullPath = Join-Path -Path $ProjectRoot -ChildPath $LegalSystemBaseDirName
$BackupDirName = "_project_refinement_backup_$(Get-Date -Format 'yyyyMMddHHmmss')"
$BackupFullPath = Join-Path -Path $ProjectRoot -ChildPath $BackupDirName

# --- Ensure Legal System Root Exists ---
if (-not (Test-Path $LegalSystemFullPath -PathType Container)) {
    Write-Error "The '$LegalSystemBaseDirName' directory was not found in '$ProjectRoot'."
    exit 1
}

# --- Create Backup Directory ---
try {
    New-Item -ItemType Directory -Path $BackupFullPath -Force | Out-Null
    Write-Host "Refinement backup directory created at '$BackupFullPath'"
}
catch {
    Write-Error "Failed to create backup directory '$BackupFullPath'. Error: $_"
    exit 1
}

# --- Function to Move Specific File to New Location (via Backup) ---
function Move-SpecificFile {
    param (
        [string]$RelativeOldPathInLegalSystem, # e.g., "base_agent.py" or "config/ontology_extraction.py"
        [string]$RelativeNewDirInLegalSystem,  # e.g., "core" or "utils"
        [string]$NewFileName = $null             # Optional: if the file should be renamed
    )

    $oldFileFullPath = Join-Path -Path $LegalSystemFullPath -ChildPath $RelativeOldPathInLegalSystem
    $targetDirFullPath = Join-Path -Path $LegalSystemFullPath -ChildPath $RelativeNewDirInLegalSystem
    $originalFileName = Split-Path -Path $oldFileFullPath -Leaf
    $finalFileName = if ($NewFileName) { $NewFileName } else { $originalFileName }
    $newFileFullPath = Join-Path -Path $targetDirFullPath -ChildPath $finalFileName

    if (-not (Test-Path $oldFileFullPath)) {
        Write-Warning "SOURCE FILE NOT FOUND, skipping: '$LegalSystemBaseDirName\$RelativeOldPathInLegalSystem'"
        # Check if it's already in the target location
        if (Test-Path $newFileFullPath) {
            Write-Host "  INFO: File '$finalFileName' already seems to be in the target directory '$LegalSystemBaseDirName\$RelativeNewDir'."
        }
        return
    }

    # 1. Ensure target directory exists
    if (-not (Test-Path $targetDirFullPath -PathType Container)) {
        try {
            New-Item -ItemType Directory -Path $targetDirFullPath -Force | Out-Null
            Write-Host "CREATED Target Directory: '$LegalSystemBaseDirName\$RelativeNewDir'"
        } catch {
            Write-Error "Failed to create target directory '$LegalSystemBaseDirName\$RelativeNewDir'. Error: $_"
            return # Cannot proceed if target dir cannot be made
        }
    }

    # 2. Move to backup first
    $backupFilePath = Join-Path -Path $BackupFullPath -ChildPath $originalFileName
    $count = 1
    $tempBackupFileName = $originalFileName
    while (Test-Path $backupFilePath) { # Avoid overwriting in backup
        $ext = [System.IO.Path]::GetExtension($tempBackupFileName)
        $nameWithoutExt = [System.IO.Path]::GetFileNameWithoutExtension($tempBackupFileName)
        $backupFilePath = Join-Path -Path $BackupFullPath -ChildPath "$nameWithoutExt`_bkp$count$ext"
        $count++
    }
    try {
        Move-Item -Path $oldFileFullPath -Destination $backupFilePath -Force
        Write-Host "MOVED to backup: '$LegalSystemBaseDirName\$RelativeOldPathInLegalSystem' -> '$($backupFilePath.Replace($ProjectRoot + "\", ""))'"
    } catch {
        Write-Error "Failed to move '$LegalSystemBaseDirName\$RelativeOldPathInLegalSystem' to backup. Error: $_"
        return
    }

    # 3. Move from backup to new final location
    try {
        Move-Item -Path $backupFilePath -Destination $newFileFullPath -Force
        Write-Host "MOVED from backup to final: '$($backupFilePath.Replace($ProjectRoot + "\", ""))' -> '$LegalSystemBaseDirName\$RelativeNewDir\$finalFileName'"
    } catch {
        Write-Error "Failed to move '$($backupFilePath.Replace($ProjectRoot + "\", ""))' to '$LegalSystemBaseDirName\$RelativeNewDir\$finalFileName'. PLEASE MOVE MANUALLY FROM BACKUP. Error: $_"
    }
}

Write-Host "`n--- Moving Core Files from '$LegalSystemBaseDirName\' root to '$LegalSystemBaseDirName\core\' ---"
Move-SpecificFile -RelativeOldPath "base_agent.py" -RelativeNewDir "core"
Move-SpecificFile -RelativeOldPath "configuration_manager.py" -RelativeNewDir "core"
Move-SpecificFile -RelativeOldPath "llm_providers.py" -RelativeNewDir "core"
Move-SpecificFile -RelativeOldPath "shared_components.py" -RelativeNewDir "core"
Move-SpecificFile -RelativeOldPath "system_initializer.py" -RelativeNewDir "core"

Write-Host "`n--- Moving Other Misplaced Files ---"
# utils/model_switcher.py -> core/model_switcher.py
Move-SpecificFile -RelativeOldPath "utils\model_switcher.py" -RelativeNewDir "core" 

# agents/ontology_extraction/*.yaml -> config/ontology_extraction/
Move-SpecificFile -RelativeOldPath "agents\ontology_extraction\entity_patterns.yaml" -RelativeNewDir "config\ontology_extraction"
Move-SpecificFile -RelativeOldPath "agents\ontology_extraction\relationship_patterns.yaml" -RelativeNewDir "config\ontology_extraction"

# config/ontology_extraction.py -> utils/ontology.py
Move-SpecificFile -RelativeOldPath "config\ontology_extraction.py" -RelativeNewDir "utils" -NewFileName "ontology.py"

# knowledge/optimized_vector_store.py (move to backup, mark for deletion - already done by previous script if it worked on this)
$optVecStorePath = Join-Path -Path $LegalSystemFullPath -ChildPath "knowledge\optimized_vector_store.py"
if (Test-Path $optVecStorePath) {
    Move-ToBackupAndSuggest -RelativeOldPath "knowledge\optimized_vector_store.py" -RelativeNewDir "knowledge\vector_store" -MarkForRemoval $true -Reason "Merged into vector_store.py"
} else {
    Write-Host "INFO: '$LegalSystemBaseDirName\knowledge\optimized_vector_store.py' not found (likely already handled/backed up)."
}


# Files that should remain in the project root (E:\A_Code_Project\)
Write-Host "`n--- Files to typically remain in Project Root '$ProjectRoot' (outside '$LegalSystemBaseDirName') ---"
Write-Host "- main.py (and main2.py if it's another entry point)"
Write-Host "- streamlit_app.py (if applicable)"
Write-Host "- __main__.py (IF it's for running the *entire project* as a module from parent of ProjectRoot. Otherwise, consider legal_ai_system/__main__.py)"
Write-Host "- requirements.txt, README.md, .gitignore, etc. (not shown in tree)"
Write-Host "- Your .txt note files are fine here for development."

Write-Host "`n--- Ensuring __init__.py in key created/targeted directories ---"
# $dirsForInitPy from previous script, adjusted to ensure targets of moves have inits
$dirsForInitPyAfterMove = @(
    "core", # Critical target
    "config",
    "config\ontology_extraction", # Critical target
    "utils" # Critical target
    # Other dirs like agents/*/* should have had their __init__.py created by previous script.
)
foreach ($dirRelToLegalSystem in $dirsForInitPyAfterMove) {
    $currentDirFullPath = Join-Path -Path $LegalSystemFullPath -ChildPath $dirRelToLegalSystem
    $initFilePath = Join-Path -Path $currentDirFullPath -ChildPath "__init__.py"

    if (-not (Test-Path $currentDirFullPath -PathType Container)) {
        New-Item -ItemType Directory -Path $currentDirFullPath -Force -ErrorAction SilentlyContinue | Out-Null
    }
    if (-not (Test-Path $initFilePath)) {
        $commentPath = $dirRelToLegalSystem -replace "\\", "/" 
        $fileContent = "# $LegalSystemBaseDirName/$commentPath/__init__.py`n# Initialization file.`n"
        New-Item -Path $initFilePath -ItemType File -Force | Out-Null
        Set-Content -Path $initFilePath -Value $fileContent -Encoding UTF8 -NoNewline -Force
        Write-Host "CREATED (or ensured): '$LegalSystemBaseDirName\$dirRelToLegalSystem\__init__.py'"
    } else {
        Write-Host "EXISTS: '$LegalSystemBaseDirName\$dirRelToLegalSystem\__init__.py'"
    }
}


Write-Host "`n--- Refinement Script Finished ---"
Write-Host "Files have been moved directly to their target locations after being backed up."
Write-Host "Please REVIEW the '$($BackupFullPath.Replace($ProjectRoot + "\", ""))' folder to ensure all intended files were moved correctly."
Write-Host "If any 'Failed to move' errors occurred for the second stage (backup to final), please move those files MANUALLY from backup."
Write-Host "After verifying, run 'list_project_structure.ps1' again and provide the output."