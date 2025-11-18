$testFiles = Get-ChildItem -Path . -Filter "test_*.py" -File
$testFiles += Get-ChildItem -Path tests -Filter "test_*.py" -File

$failedTests = @()

foreach ($file in $testFiles) {

    Write-Host "Running $($file.Name)"

    & python $file.FullName

    if ($LASTEXITCODE -ne 0) {

        Write-Host "FAILED: $($file.Name)"

        $failedTests += $file.Name

    } else {

        Write-Host "PASSED: $($file.Name)"

    }

}

if ($failedTests.Count -gt 0) {

    Write-Host "Failed tests: $($failedTests -join ', ')"

} else {

    Write-Host "All tests passed."

}