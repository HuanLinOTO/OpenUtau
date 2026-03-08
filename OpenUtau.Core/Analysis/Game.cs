using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.Analysis.Game {

    class GameConfig {
        [JsonPropertyName("samplerate")]
        public int SampleRate { get; set; } = 44100;

        [JsonPropertyName("timestep")]
        public float Timestep { get; set; } = 0.01f;

        [JsonPropertyName("languages")]
        public Dictionary<string, int>? Languages { get; set; }

        [JsonPropertyName("loop")]
        public bool Loop { get; set; } = true;

        [JsonPropertyName("embedding_dim")]
        public int EmbeddingDim { get; set; } = 256;
    }

    public class Game : IDisposable {
        InferenceSession encoderSession;
        InferenceSession segmenterSession;
        InferenceSession estimatorSession;
        InferenceSession bd2durSession;
        GameConfig config;
        string Location;
        private bool disposedValue;

        // D3PM sampling parameters
        int samplingSteps = 8;
        float boundaryThreshold = 0.3f;
        int boundaryRadius = 2;
        float scoreThreshold = 0.2f;

        struct GameResult {
            public float[] durations;
            public bool[] presence;
            public float[] scores;
            public bool[] maskN;
            public int N;
        }

        public Game() {
            Location = Path.Combine(PathManager.Inst.DependencyPath, "game");
            string configPath = Path.Combine(Location, "config.json");
            if (!File.Exists(configPath)) {
                throw new MessageCustomizableException(
                    "GAME not found",
                    "<translate:errors.failed.transcribe.game>",
                    new FileNotFoundException(),
                    false,
                    new string[] { "https://github.com/openvpi/GAME/releases/tag/v1.0.1" }
                );
            }

            var jsonText = File.ReadAllText(configPath, System.Text.Encoding.UTF8);
            config = JsonSerializer.Deserialize<GameConfig>(jsonText)
                     ?? throw new InvalidOperationException("Failed to parse GAME config.json");

            encoderSession = Onnx.getInferenceSession(
                Path.Combine(Location, "encoder.onnx"));
            segmenterSession = Onnx.getInferenceSession(
                Path.Combine(Location, "segmenter.onnx"));
            estimatorSession = Onnx.getInferenceSession(
                Path.Combine(Location, "estimator.onnx"));
            bd2durSession = Onnx.getInferenceSession(
                Path.Combine(Location, "bd2dur.onnx"));
        }

        /// <summary>
        /// Run encoder: waveform -> x_seg, x_est, maskT
        /// </summary>
        private (float[] x_seg, float[] x_est, bool[] maskT, int T)
        RunEncoder(float[] samples) {
            float duration = (float)samples.Length / config.SampleRate;

            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("waveform",
                    new DenseTensor<float>(samples, new int[] { 1, samples.Length })),
                NamedOnnxValue.CreateFromTensor("duration",
                    new DenseTensor<float>(new float[] { duration }, new int[] { 1 })),
            };

            using var outputs = encoderSession.Run(inputs);

            var x_seg_tensor = outputs.First(o => o.Name == "x_seg").AsTensor<float>();
            var x_est_tensor = outputs.First(o => o.Name == "x_est").AsTensor<float>();
            var maskT_tensor = outputs.First(o => o.Name == "maskT").AsTensor<bool>();

            int T = x_seg_tensor.Dimensions[1];

            return (
                x_seg_tensor.ToArray(),
                x_est_tensor.ToArray(),
                maskT_tensor.ToArray(),
                T
            );
        }

        /// <summary>
        /// Run a single segmenter step (D3PM sampling iteration)
        /// </summary>
        private bool[] RunSegmenterStep(
            float[] x_seg, int T, int C,
            bool[] knownBoundaries, bool[] prevBoundaries,
            float t, bool[] maskT) {

            var inputs = new List<NamedOnnxValue>();
            inputs.Add(NamedOnnxValue.CreateFromTensor("x_seg",
                new DenseTensor<float>(x_seg, new int[] { 1, T, C })));

            // Add language input if model supports it
            if (segmenterSession.InputNames.Contains("language")) {
                inputs.Add(NamedOnnxValue.CreateFromTensor("language",
                    new DenseTensor<long>(new long[] { 0 }, new int[] { 1 }))); // 0 = universal
            }

            inputs.Add(NamedOnnxValue.CreateFromTensor("known_boundaries",
                new DenseTensor<bool>(knownBoundaries, new int[] { 1, T })));

            // D3PM mode inputs
            if (config.Loop && segmenterSession.InputNames.Contains("prev_boundaries")) {
                inputs.Add(NamedOnnxValue.CreateFromTensor("prev_boundaries",
                    new DenseTensor<bool>(prevBoundaries, new int[] { 1, T })));
                inputs.Add(NamedOnnxValue.CreateFromTensor("t",
                    new DenseTensor<float>(new float[] { t }, new int[] { 1 })));
            }

            inputs.Add(NamedOnnxValue.CreateFromTensor("maskT",
                new DenseTensor<bool>(maskT, new int[] { 1, T })));
            inputs.Add(NamedOnnxValue.CreateFromTensor("threshold",
                new DenseTensor<float>(new float[] { boundaryThreshold }, Array.Empty<int>())));
            inputs.Add(NamedOnnxValue.CreateFromTensor("radius",
                new DenseTensor<long>(new long[] { boundaryRadius }, Array.Empty<int>())));

            using var outputs = segmenterSession.Run(inputs);
            return outputs.First(o => o.Name == "boundaries").AsTensor<bool>().ToArray();
        }

        /// <summary>
        /// Run full segmentation with D3PM iterative sampling loop
        /// </summary>
        private bool[] RunSegmentation(float[] x_seg, int T, int C, bool[] maskT) {
            bool[] knownBoundaries = new bool[T]; // all false = no known boundaries
            bool[] boundaries = (bool[])knownBoundaries.Clone();

            if (config.Loop) {
                // D3PM sampling: iterate from t=0 (full noise) towards t=1 (data)
                float step = 1.0f / samplingSteps;
                for (int i = 0; i < samplingSteps; i++) {
                    float t = i * step;
                    boundaries = RunSegmenterStep(
                        x_seg, T, C, knownBoundaries, boundaries, t, maskT);
                }
            } else {
                // Completion mode: single pass
                boundaries = RunSegmenterStep(
                    x_seg, T, C, knownBoundaries, knownBoundaries, 0f, maskT);
            }

            return boundaries;
        }

        /// <summary>
        /// Run bd2dur: boundaries -> durations (seconds) + maskN
        /// </summary>
        private (float[] durations, bool[] maskN, int N)
        RunBd2Dur(bool[] boundaries, bool[] maskT, int T) {
            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("boundaries",
                    new DenseTensor<bool>(boundaries, new int[] { 1, T })),
                NamedOnnxValue.CreateFromTensor("maskT",
                    new DenseTensor<bool>(maskT, new int[] { 1, T })),
            };

            using var outputs = bd2durSession.Run(inputs);
            var durations_tensor = outputs.First(o => o.Name == "durations").AsTensor<float>();
            var maskN_tensor = outputs.First(o => o.Name == "maskN").AsTensor<bool>();
            int N = durations_tensor.Dimensions[1];

            return (durations_tensor.ToArray(), maskN_tensor.ToArray(), N);
        }

        /// <summary>
        /// Run estimator: predict note presence and pitch scores
        /// </summary>
        private (bool[] presence, float[] scores)
        RunEstimator(float[] x_est, bool[] boundaries, bool[] maskT, bool[] maskN,
                     int T, int C, int N) {
            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("x_est",
                    new DenseTensor<float>(x_est, new int[] { 1, T, C })),
                NamedOnnxValue.CreateFromTensor("boundaries",
                    new DenseTensor<bool>(boundaries, new int[] { 1, T })),
                NamedOnnxValue.CreateFromTensor("maskT",
                    new DenseTensor<bool>(maskT, new int[] { 1, T })),
                NamedOnnxValue.CreateFromTensor("maskN",
                    new DenseTensor<bool>(maskN, new int[] { 1, N })),
                NamedOnnxValue.CreateFromTensor("threshold",
                    new DenseTensor<float>(new float[] { scoreThreshold }, Array.Empty<int>())),
            };

            using var outputs = estimatorSession.Run(inputs);
            return (
                outputs.First(o => o.Name == "presence").AsTensor<bool>().ToArray(),
                outputs.First(o => o.Name == "scores").AsTensor<float>().ToArray()
            );
        }

        /// <summary>
        /// Full GAME inference pipeline: waveform -> notes
        /// 1. Encoder: waveform -> latent features
        /// 2. Segmenter: D3PM iterative boundary prediction
        /// 3. Bd2Dur: boundaries -> note durations
        /// 4. Estimator: note pitch estimation
        /// </summary>
        GameResult Analyze(float[] samples) {
            // 1. Encoder
            var (x_seg, x_est, maskT, T) = RunEncoder(samples);
            int C = config.EmbeddingDim;

            // 2. Segmentation (D3PM loop)
            var boundaries = RunSegmentation(x_seg, T, C, maskT);

            // 3. Boundaries to durations
            var (durations, maskN, N) = RunBd2Dur(boundaries, maskT, T);

            // 4. Estimation
            var (presence, scores) = RunEstimator(x_est, boundaries, maskT, maskN, T, C, N);

            return new GameResult {
                durations = durations,
                presence = presence,
                scores = scores,
                maskN = maskN,
                N = N,
            };
        }

        private float[] ToMono(float[] stereoSamples, int channels) {
            if (channels == 1) {
                return stereoSamples;
            }
            float[] monoSamples = new float[stereoSamples.Length / channels];
            for (int i = 0; i < monoSamples.Length; i++) {
                monoSamples[i] = stereoSamples[(i * channels)..((i + 1) * channels - 1)].Average();
            }
            return monoSamples;
        }

        /// <summary>
        /// Transcribe a wave part into a voice part using GAME model.
        /// Slices audio by silence, runs GAME inference on each chunk,
        /// and assembles notes into a UVoicePart.
        /// </summary>
        public UVoicePart Transcribe(UProject project, UWavePart wavePart, Action<int> progress) {
            var monoSamples = ToMono(wavePart.Samples, wavePart.channels);
            var chunks = Some.AudioSlicer.Slice(monoSamples);
            var part = new UVoicePart();
            part.position = wavePart.position;
            part.Duration = wavePart.Duration;
            var timeAxis = project.timeAxis;
            double partOffsetMs = timeAxis.TickPosToMsPos(wavePart.position);
            double currMs = partOffsetMs;

            int wavPosS = 0;
            foreach (var chunk in chunks) {
                wavPosS = (int)(chunk.offsetMs / 1000);
                progress.Invoke(wavPosS);
                var result = Analyze(chunk.samples);

                double chunkOffsetMs = chunk.offsetMs + partOffsetMs;
                currMs = chunkOffsetMs;
                for (int index = 0; index < result.N; index++) {
                    if (!result.maskN[index]) break; // end of valid notes (padding)

                    var noteDurMs = result.durations[index] * 1000;
                    if (result.presence[index]) {
                        var posTick = timeAxis.MsPosToTickPos(currMs);
                        var durTick = timeAxis.MsPosToTickPos(currMs + noteDurMs) - posTick;
                        if (durTick > 0) {
                            var note = project.CreateNote(
                                (int)Math.Round(result.scores[index]),
                                posTick - wavePart.position,
                                durTick
                            );
                            part.notes.Add(note);
                        }
                    }
                    currMs += noteDurMs;
                }
            }

            var endTick = timeAxis.MsPosToTickPos(currMs);
            if (endTick > part.End) {
                part.Duration = endTick - part.position;
            }
            return part;
        }

        protected virtual void Dispose(bool disposing) {
            if (!disposedValue) {
                if (disposing) {
                    encoderSession?.Dispose();
                    segmenterSession?.Dispose();
                    estimatorSession?.Dispose();
                    bd2durSession?.Dispose();
                }
                disposedValue = true;
            }
        }

        public void Dispose() {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
