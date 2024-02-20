﻿using System.Diagnostics;
using System.Threading.Tasks.Dataflow;
using Extractor.Commands;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkyRemoval;
using Spectre.Console;
using TreeBasedCli;
namespace Extractor.Handlers;

public class MaskSkyCommandHandler : ILeafCommandHandler<MaskSkyCommand.MaskSkyArguments>
{
    private readonly List<SkyRemovalModel> _models = [];

    public async Task HandleAsync(MaskSkyCommand.MaskSkyArguments arguments, LeafCommand executedCommand)
    {


        var stopwatch = Stopwatch.StartNew();


        var finalMaskCount = 0;

        const long maxMemory = 2000000000;
        await AnsiConsole.Progress()
            .AutoClear(false) // Do not remove the task list when done
            .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new ElapsedTimeColumn(), new RemainingTimeColumn(), new SpinnerColumn())
            .StartAsync(async ctx =>
            {
                var loadingModelProgressTask = ctx.AddTask("[green]Loading Sky Removal Model[/]", true);
                var frameListProgressTask = ctx.AddTask("[green]Loading Frame List[/]", true);

                var loadImageProgressTask = ctx.AddTask("[green]Loading Images[/]");
                var prepareImageProgressTask = ctx.AddTask("[green]Preparing Images[/]");
                var runningModelProgressTask = ctx.AddTask("[green]Running Model[/]");
                var postProcessingProgressTask = ctx.AddTask("[green]Post-processing Images[/]");

                var savingProgressTask = ctx.AddTask("[green]Saving Masks[/]");


                loadingModelProgressTask.IsIndeterminate = true;
                
                
                
                foreach (var gpu in Enumerable.Range(0, arguments.ProcessorCount))
                {
                    _models.Add(await SkyRemovalModel.CreateAsync(arguments.Engine, gpu));
                }
                
                
            
                loadingModelProgressTask.Complete();


                Directory.CreateDirectory(arguments.OutputPath);

                frameListProgressTask.IsIndeterminate = true;


                var frameList = GenerateFrameList(arguments.InputPath, arguments.InputType).ToList();

                finalMaskCount = frameList.Count;

                frameListProgressTask.Complete();


                prepareImageProgressTask.MaxValue(frameList.Count);
                runningModelProgressTask.MaxValue(frameList.Count);
                postProcessingProgressTask.MaxValue(frameList.Count);
                savingProgressTask.MaxValue(frameList.Count);

                var processorCount = Environment.ProcessorCount;


                var parallelDataFlowBlockExecutionOptions = new ExecutionDataflowBlockOptions
                {
                    MaxDegreeOfParallelism = processorCount,
                    BoundedCapacity = processorCount * 2,
                    EnsureOrdered = false,
                };

                var serialDataFlowBlockExecutionOptions = new ExecutionDataflowBlockOptions
                {
                    MaxDegreeOfParallelism = 1,
                    EnsureOrdered = false,
                    BoundedCapacity = 1
                };

                var dataflowBlockOptions = new DataflowBlockOptions
                {
                    EnsureOrdered = false,
                    BoundedCapacity = processorCount * 4
                };
                
                var dataflowLinkOptions = new DataflowLinkOptions
                {
                    PropagateCompletion = true,
                    Append = true,
                };


                var inputBufferBlock = new BufferBlock<string>(dataflowBlockOptions);


                var loadImageBlock = new TransformBlock<string, ImageContainer>(async path =>
                {
                    var img = await Image.LoadAsync(path);

                    loadImageProgressTask.Increment(1);

                    return new ImageContainer(img, path);
                }, parallelDataFlowBlockExecutionOptions);


                var imageBufferBlock = new BufferBlock<ImageContainer>(dataflowBlockOptions);


                var createTensorBlock = new TransformBlock<ImageContainer, TensorContainer>(imageContainer =>
                {
                    var tensor = _models[0].PrepareImage(imageContainer.Image);
                    var container = new TensorContainer(tensor, imageContainer);

                    prepareImageProgressTask.Increment(1);

                    return container;
                }, parallelDataFlowBlockExecutionOptions);

                var modelBufferBlock = new BufferBlock<TensorContainer>(dataflowBlockOptions);

                
                
                var postMaskingBufferBlock = new BufferBlock<TensorContainer>(dataflowBlockOptions);

                // for each model, create a new block that is connected to the modelBufferBlock and then to a join block

                foreach (var skyRemovalModel in _models)
                {
                    var generateMaskTensorBlock = new TransformBlock<TensorContainer, TensorContainer>(tensorContainer =>
                    {
                        var result = skyRemovalModel.RunModel(tensorContainer.Tensor);

                        runningModelProgressTask.Increment(1);

                        return new TensorContainer(result, tensorContainer.OriginalImageContainer);
                    }, serialDataFlowBlockExecutionOptions);
                    
                    modelBufferBlock.LinkTo(generateMaskTensorBlock, dataflowLinkOptions);
                    generateMaskTensorBlock.LinkTo(postMaskingBufferBlock, dataflowLinkOptions);
                }
                
                
             

                var postProcessTensorBlock = new TransformBlock<TensorContainer, ImageContainer>(container =>
                {
                    var output = _models[0].ProcessModelResults(container.OriginalImageContainer.Image, container.Tensor);
                    postProcessingProgressTask.Increment(1);

                    return new ImageContainer(output, container.OriginalImageContainer.Path);
                }, parallelDataFlowBlockExecutionOptions);

                var saveBuffer = new BufferBlock<ImageContainer>(dataflowBlockOptions);

                var saveImageBlock = new ActionBlock<ImageContainer>(async container =>
                {
                    var path = Path.Combine(arguments.OutputPath, Path.GetFileNameWithoutExtension(container.Path) + "_mask" + Path.GetExtension(container.Path));
                    await container.Image.SaveAsync(path);
                    savingProgressTask.Increment(1);
                }, parallelDataFlowBlockExecutionOptions);


              

                inputBufferBlock.LinkTo(loadImageBlock, dataflowLinkOptions);
                loadImageBlock.LinkTo(imageBufferBlock, dataflowLinkOptions);
                imageBufferBlock.LinkTo(createTensorBlock, dataflowLinkOptions);
                createTensorBlock.LinkTo(modelBufferBlock, dataflowLinkOptions);
                postMaskingBufferBlock.LinkTo(postProcessTensorBlock, dataflowLinkOptions);
                postProcessTensorBlock.LinkTo(saveBuffer, dataflowLinkOptions);
                saveBuffer.LinkTo(saveImageBlock, dataflowLinkOptions);

                foreach (var frame in frameList)
                {

                    while (true)
                    {
                        var memory = GC.GetTotalMemory(forceFullCollection: true); // Returns the current memory usage in bytes

                        if ((maxMemory - memory) >= 30000000) // Checks if there's at least 30 MB free within your set limit
                        {
                            await inputBufferBlock.SendAsync(frame);
                            break; // Once the SendAsync is executed, break from the while loop
                        }
                        Thread.Sleep(500); // If there's not enough memory, wait for a bit before checking again
                    }
                }

                inputBufferBlock.Complete();
                await saveImageBlock.Completion;

            });

        stopwatch.Stop();

        AnsiConsole.MarkupLineInterpolated($"Finished generating {finalMaskCount} masks in {Math.Round(stopwatch.ElapsedMilliseconds * 0.001, 2)} seconds.");
    }

    private static IEnumerable<string> GenerateFrameList(string inputPath, PathType inputType)
    {
        if (inputType.HasFlag(PathType.File))
        {
            yield return inputPath;
        }

        if (inputType.HasFlag(PathType.Directory))
        {
            foreach (var directory in Directory.EnumerateFiles(inputPath, "*", SearchOption.AllDirectories))
            {
                yield return directory;
            }
        }
    }

    private class TensorContainer
    {
        public readonly ImageContainer OriginalImageContainer;
        public readonly Tensor<float> Tensor;
        public TensorContainer(Tensor<float> tensor, ImageContainer originalImageContainer)
        {
            Tensor = tensor;
            OriginalImageContainer = originalImageContainer;
        }
    }

    private class ImageContainer
    {
        public readonly Image Image;
        public readonly string Path;
        public ImageContainer(Image image, string path)
        {
            Image = image;
            Path = path;
        }
    }
}
