using System.Numerics;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
namespace SkyRemoval
{
    public class SkyRemovalModel
    {
        private const int ModelWidth = 384;
        private const int ModelHeight = 384;

        private static string? _modelPath;


        private static SkyRemovalModel? _instance;
        private static readonly SemaphoreSlim SetupSemaphore = new(1, 1);

        private readonly InferenceSession _session;

        // Make the constructor private to prevent direct construction calls.
        private SkyRemovalModel(string modelPath, ExecutionEngine engine, int gpuId)
        {

            if (engine != ExecutionEngine.Auto)
            {
                var sessionOptions = engine switch
                {
                    ExecutionEngine.CPU => CreateCpuSessionOptions(),
                    ExecutionEngine.CUDA => CreateCudaSession(),
                    ExecutionEngine.TensorRT => SessionOptions.MakeSessionOptionWithTensorrtProvider(),
                    ExecutionEngine.DirectML => CreateDirectMl(),
                    ExecutionEngine.Auto => throw new ArgumentOutOfRangeException(nameof(engine), engine, null),
                    _ => throw new ArgumentOutOfRangeException(nameof(engine), engine, null)
                };

                _session = new InferenceSession(modelPath, sessionOptions);
            }
            else
            {
                _session = CreateInferenceSession(
                    modelPath,
                    () => CreateTensorRt(gpuId),
                    () => CreateCudaSession(gpuId),
                    () => CreateDirectMl(gpuId),
                    CreateCpuSessionOptions
                );

            }
        }

        private static InferenceSession CreateInferenceSession(string modelPath, params Func<SessionOptions>[] sessionOptionFactories)
        {
            foreach (var sessionOptionFactory in sessionOptionFactories)
            {
                try
                {
                    var inferenceSession = new InferenceSession(modelPath, sessionOptionFactory());
                    return inferenceSession;
                }
                catch (Exception)
                {
                    // Ignore
                }
            }

            throw new Exception("Failed to create an InferenceSession with any provider");
        }

        private static SessionOptions CreateCudaSession(int gpuId = 0, int memoryLimitGb = 6)
        {

            var cudaProviderOptions = new OrtCUDAProviderOptions(); // Dispose this finally

            var providerOptionsDict = new Dictionary<string, string>
            {
                ["gpu_mem_limit"] = "4294967296",
                ["arena_extend_strategy"] = "kSameAsRequested",
                ["cudnn_conv_algo_search"] = "EXHAUSTIVE",
                ["do_copy_in_default_stream"] = "1",
                ["cudnn_conv_use_max_workspace"] = "1",
                ["cudnn_conv1d_pad_to_nc1d"] = "1"
            };


            cudaProviderOptions.UpdateOptions(providerOptionsDict);

            return SessionOptions.MakeSessionOptionWithCudaProvider(gpuId);
        }


        private static SessionOptions CreateCpuSessionOptions()
        {
            var sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sessionOptions.EnableMemoryPattern = true;


            return sessionOptions;
        }

        private static SessionOptions CreateTensorRt(int gpuId = 0)
        {
            var providerOptionsDict = new Dictionary<string, string>
            {
                ["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = "4294967296",
                ["ORT_TENSORRT_FP16_ENABLE"] = "1",
                ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1",
                ["ORT_TENSORRT_BUILDER_OPTIMIZATION_LEVEL"] = "5",
                ["ORT_TENSORRT_CONTEXT_MEMORY_SHARING_ENABLE"] = "1"


            };





            var options = SessionOptions.MakeSessionOptionWithTensorrtProvider(gpuId);
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            return options;
        }

        private static SessionOptions CreateDirectMl(int gpuId = 0)
        {
            var sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.AppendExecutionProvider_DML(gpuId);

            return sessionOptions;
        }

        public static async Task<SkyRemovalModel> CreateAsync(ExecutionEngine engine = ExecutionEngine.Auto, int gpuId = 0)
        {
            if (_instance != null)
                return _instance;

            await SetupSemaphore.WaitAsync();

            try
            {
                if (_instance != null)
                {
                    return _instance;
                }

                if (string.IsNullOrEmpty(_modelPath))
                    _modelPath = await ModelSetup.SetupModel();

                if (string.IsNullOrEmpty(_modelPath))
                {
                    throw new Exception("Model not setup");
                }

                _instance = new SkyRemovalModel(_modelPath, engine, gpuId);
            }
            finally
            {
                SetupSemaphore.Release();
            }

            return _instance;
        }

        public Image<Rgb24> Run(Image image)
        {
            var npImg = PrepareImage(image);

            var output = RunModel(npImg);

            return ProcessModelResults(image, output);
        }
        public Image<Rgb24> ProcessModelResults(Image originalImage, Tensor<float> outputTensor)
        {
            var height = outputTensor.Dimensions[2];
            var width = outputTensor.Dimensions[3];

            var processedImage = new Image<Rgb24>(Configuration.Default, width, height);

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var pixelValue = (byte)(outputTensor[0, 0, y, x] * 255); // Using outputTensor instead of ndArray
                    processedImage[x, y] = new Rgb24(pixelValue, pixelValue, pixelValue);
                }
            }

            var originalWidth = originalImage.Width;
            var originalHeight = originalImage.Height;

            processedImage.Mutate(x =>
            {
                x.ProcessPixelRowsAsVector4(pixelRow => Clip(pixelRow, 0f, 1f));

                x.Resize(new ResizeOptions
                {
                    Size = new Size(originalWidth, originalHeight),
                    Mode = ResizeMode.Stretch,
                    Sampler = KnownResamplers.Lanczos3
                });
            });

            var inputSrc = originalImage.CloneAs<Rgb24>().ConvertToEmguCv().Mat;
            var outputSrc = processedImage.ConvertToEmguCv().Mat;

            var dst = new Mat();


            XImgprocInvoke.GuidedFilter(outputSrc, inputSrc, dst, 20, 0.01f);

            processedImage = dst.ToImage<Bgr, byte>().ConvertToImageSharp();


            processedImage.Mutate(x =>
            {
                x.ProcessPixelRowsAsVector4(pixelRow => Clip(pixelRow, 0f, 1f));
                x.Grayscale();
                x.BinaryThreshold(0.5f);
                x.Invert();
            });
            return processedImage;
        }
        public DenseTensor<float> PrepareImage(Image image)
        {
            var processingImage = image.CloneAs<Rgb24>();

            processingImage.Mutate(x =>
            {
                x.ProcessPixelRowsAsVector4(NormalizePixelRow);
                x.Resize(new ResizeOptions
                {
                    Size = new Size(ModelWidth, ModelHeight),
                    Mode = ResizeMode.Stretch,
                    Sampler = KnownResamplers.Box
                });

            });

            var ndArray = ImageToNdArray(processingImage);

            var npImg = TransposeExpandNdArray(ndArray);

            return NdArrayToDenseTensor(npImg);
        }
        private static void Clip(Span<Vector4> pixelRow, float min = 0, float max = 1)
        {

            foreach (ref var pixel in pixelRow)
            {
                pixel.X = Math.Clamp(pixel.X, min, max);
                pixel.Y = Math.Clamp(pixel.Y, min, max);
                pixel.Z = Math.Clamp(pixel.Z, min, max);
                pixel.W = Math.Clamp(pixel.W, min, max);
            }
        }

        private static void NormalizePixelRow(Span<Vector4> pixelRow)
        {
            foreach (ref var pixel in pixelRow)
            {
                pixel /= pixel.W;
                pixel /= 255.0f; // Convert from [0,255] to [0,1]
            }
        }



        private static NDArray TransposeExpandNdArray(NDArray npImg)
        {
            npImg = npImg.transpose(new[]
            {
                2,
                0,
                1
            });


            return npImg.reshape(1, npImg.shape[0], npImg.shape[1], npImg.shape[2]);
        }

        public Tensor<float> RunModel(Tensor<float> tensor)
        {
            if (_modelPath == null)
            {
                throw new Exception("Model not setup");
            }

            var inputName = _session.InputNames[0];

            using var onnxOutput = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(inputName, tensor)
            });

            var outputTensor = onnxOutput.First().AsTensor<float>().Clone();


            return outputTensor;
        }


        private static DenseTensor<float> NdArrayToDenseTensor(NDArray npImg)
        {
            var data = npImg.Data<float>().ToArray();
            var tensor = new DenseTensor<float>(data, npImg.shape);
            return tensor;
        }



        private static NDArray ImageToNdArray(Image<Rgb24> image)
        {
            var height = image.Height;
            var width = image.Width;
            var npImg = new NDArray(typeof(float), new Shape(height, width, 3));

            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    var pixel = image[j, i]; // Access image pixels in (width, height) order
                    npImg[i, j, 0] = pixel.R / 255f;
                    npImg[i, j, 1] = pixel.G / 255f;
                    npImg[i, j, 2] = pixel.B / 255f;
                }
            }

            return npImg;
        }


        private static Image<Rgb24> NdArrayToImage(NDArray ndArray)
        {
            if (ndArray.ndim != 3 || ndArray.shape[2] < 3)
            {
                throw new ArgumentException("Invalid NDArray shape. Expecting a 3D array with at least 3 channels in the third dimension.");
            }

            var height = ndArray.shape[0];
            var width = ndArray.shape[1];
            var img = new Image<Rgb24>(Configuration.Default, width, height);

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    img[x, y] = new Rgb24(
                        (byte)(ndArray[y, x, 0] * 255),
                        (byte)(ndArray[y, x, 1] * 255),
                        (byte)(ndArray[y, x, 2] * 255));
                }
            }

            return img;
        }
    }
}
