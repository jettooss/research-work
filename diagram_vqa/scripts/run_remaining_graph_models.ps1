$ErrorActionPreference = "Stop"

$python = "C:\Users\Jet\Desktop\data\.venv\Scripts\python.exe"
$runner = "C:\Users\Jet\Desktop\data\diagram_vqa\scripts\run_model_matrix.py"
$output = "C:\Users\Jet\Desktop\data\diagram_vqa\runs\model_matrix"

$jobs = @(
    @{ Dataset = "infographicvqa"; Model = "sam2_graph_transformer" },
    @{ Dataset = "docvqa"; Model = "graphcolbert" },
    @{ Dataset = "infographicvqa"; Model = "graphcolbert" },
    @{ Dataset = "ai2d"; Model = "graphcolbert" }
)

foreach ($job in $jobs) {
    & $python $runner `
        --dataset $job.Dataset `
        --model $job.Model `
        --split test `
        --seed 42 `
        --output-dir $output `
        --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $($job.Dataset)/$($job.Model), exit code $LASTEXITCODE"
    }
}
