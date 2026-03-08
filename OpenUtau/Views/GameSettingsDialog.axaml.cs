using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using OpenUtau.Core;
using ReactiveUI;

namespace OpenUtau.App.Views {

    public class GameModelInfo {
        public string DisplayName { get; set; } = "";
        public string Path { get; set; } = "";
        public Dictionary<string, int>? Languages { get; set; }
        public float Timestep { get; set; } = 0.01f;
        public override string ToString() => DisplayName;
    }

    public class GameLanguageInfo {
        public string DisplayName { get; set; } = "";
        public int Id { get; set; } = 0;
        public override string ToString() => DisplayName;
    }

    public class GameDialogResult {
        public string ModelPath { get; set; } = "";
        public int SamplingSteps { get; set; } = 8;
        public float T0 { get; set; } = 0.0f;
        public float BoundaryThreshold { get; set; } = 0.3f;
        public float BoundaryRadiusSeconds { get; set; } = 0.02f;
        public float ScoreThreshold { get; set; } = 0.2f;
        public int LanguageId { get; set; } = 0;
        public bool ForceCpu { get; set; } = false;
    }

    public partial class GameSettingsDialog : Window {
        public Action<GameDialogResult>? onFinish;
        private List<GameModelInfo> models = new();

        public GameSettingsDialog() {
            InitializeComponent();
            LoadModels();
            BindSliders();
            // Restore persisted ForceCpu preference
            ForceCpuCheckBox.IsChecked = Core.Util.Preferences.Default.GameForceCpu;
        }

        private void LoadModels() {
            var depPath = PathManager.Inst.DependencyPath;
            if (!Directory.Exists(depPath)) return;

            var dirs = Directory.GetDirectories(depPath)
                .Where(d => {
                    var name = System.IO.Path.GetFileName(d).ToLowerInvariant();
                    return name == "game" || name.StartsWith("game-");
                })
                .Where(d => File.Exists(System.IO.Path.Combine(d, "config.json")))
                .OrderBy(d => System.IO.Path.GetFileName(d))
                .ToList();

            foreach (var dir in dirs) {
                var dirName = System.IO.Path.GetFileName(dir);
                string displayName;
                if (dirName.Equals("game", StringComparison.OrdinalIgnoreCase)) {
                    displayName = "GAME (Default)";
                } else if (dirName.StartsWith("game-", StringComparison.OrdinalIgnoreCase)) {
                    var suffix = dirName.Substring(5);
                    displayName = "GAME " + char.ToUpper(suffix[0]) + suffix.Substring(1);
                } else {
                    displayName = dirName;
                }

                try {
                    var configText = File.ReadAllText(
                        System.IO.Path.Combine(dir, "config.json"),
                        System.Text.Encoding.UTF8);
                    using var doc = JsonDocument.Parse(configText);
                    var root = doc.RootElement;

                    Dictionary<string, int>? languages = null;
                    if (root.TryGetProperty("languages", out var langProp)) {
                        languages = new Dictionary<string, int>();
                        foreach (var kv in langProp.EnumerateObject()) {
                            languages[kv.Name] = kv.Value.GetInt32();
                        }
                    }

                    float timestep = 0.01f;
                    if (root.TryGetProperty("timestep", out var tsProp)) {
                        timestep = tsProp.GetSingle();
                    }

                    models.Add(new GameModelInfo {
                        DisplayName = displayName,
                        Path = dir,
                        Languages = languages,
                        Timestep = timestep,
                    });
                } catch {
                    // skip invalid config
                }
            }

            ModelCombo.ItemsSource = models;
            if (models.Count > 0) {
                ModelCombo.SelectedIndex = 0;
            }
        }

        private void BindSliders() {
            this.WhenAnyValue(d => d.StepsSlider.Value)
                .Subscribe(v => StepsText.Text = ((int)v).ToString());
            this.WhenAnyValue(d => d.T0Slider.Value)
                .Subscribe(v => T0Text.Text = v.ToString("F2"));
            this.WhenAnyValue(d => d.SegThresholdSlider.Value)
                .Subscribe(v => SegThresholdText.Text = v.ToString("F2"));
            this.WhenAnyValue(d => d.SegRadiusSlider.Value)
                .Subscribe(v => SegRadiusText.Text = v.ToString("F3"));
            this.WhenAnyValue(d => d.EstThresholdSlider.Value)
                .Subscribe(v => EstThresholdText.Text = v.ToString("F2"));
        }

        private void OnModelChanged(object? sender, SelectionChangedEventArgs e) {
            if (ModelCombo.SelectedItem is GameModelInfo model) {
                var langItems = new List<GameLanguageInfo>();
                langItems.Add(new GameLanguageInfo {
                    DisplayName = ThemeManager.GetString("game.settings.language.auto"),
                    Id = 0,
                });
                if (model.Languages != null) {
                    foreach (var kv in model.Languages.OrderBy(x => x.Value)) {
                        langItems.Add(new GameLanguageInfo {
                            DisplayName = kv.Key,
                            Id = kv.Value,
                        });
                    }
                }
                LanguageCombo.ItemsSource = langItems;
                LanguageCombo.SelectedIndex = 0;
            }
        }

        private void OkButtonClick(object? sender, RoutedEventArgs e) {
            Finish();
        }

        private void CancelButtonClick(object? sender, RoutedEventArgs e) {
            Close();
        }

        private void Finish() {
            if (ModelCombo.SelectedItem is not GameModelInfo model) {
                Close();
                return;
            }

            int languageId = 0;
            if (LanguageCombo.SelectedItem is GameLanguageInfo lang) {
                languageId = lang.Id;
            }

            bool forceCpu = ForceCpuCheckBox.IsChecked == true;
            // Persist the ForceCpu preference
            Core.Util.Preferences.Default.GameForceCpu = forceCpu;
            Core.Util.Preferences.Save();

            var result = new GameDialogResult {
                ModelPath = model.Path,
                SamplingSteps = (int)StepsSlider.Value,
                T0 = (float)T0Slider.Value,
                BoundaryThreshold = (float)SegThresholdSlider.Value,
                BoundaryRadiusSeconds = (float)SegRadiusSlider.Value,
                ScoreThreshold = (float)EstThresholdSlider.Value,
                LanguageId = languageId,
                ForceCpu = forceCpu,
            };

            onFinish?.Invoke(result);
            Close();
        }

        protected override void OnKeyDown(KeyEventArgs e) {
            if (e.Key == Key.Escape) {
                e.Handled = true;
                Close();
            } else if (e.Key == Key.Enter) {
                e.Handled = true;
                Finish();
            } else {
                base.OnKeyDown(e);
            }
        }
    }
}
