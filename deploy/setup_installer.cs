// Standalone Windows GUI Installer for Alethic Agent (no Python or PyTorch required).
// Built by deploy\21_build_agent_exe.bat using .NET Framework 4 csc.exe (C# 5 compatible).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

static class Brand
{
    public static readonly Color Bg = Color.FromArgb(20, 19, 16);
    public static readonly Color Card = Color.FromArgb(28, 27, 23);
    public static readonly Color InputBg = Color.FromArgb(36, 34, 29);
    public static readonly Color Border = Color.FromArgb(58, 55, 48);
    public static readonly Color Amber = Color.FromArgb(255, 176, 32);
    public static readonly Color Cream = Color.FromArgb(245, 233, 200);
    public static readonly Color Muted = Color.FromArgb(163, 158, 138);
    public static readonly Color Blue = Color.FromArgb(79, 107, 255);
    public static readonly Color Green = Color.FromArgb(156, 203, 139);
    public static readonly Color Red = Color.FromArgb(255, 111, 150);

    public static void DrawDiscs(Graphics g, float cx, float cy, float r)
    {
        g.SmoothingMode = SmoothingMode.AntiAlias;
        float offset = r * 0.56f;
        RectangleF left = new RectangleF(cx - offset - r, cy - r, r * 2, r * 2);
        RectangleF right = new RectangleF(cx + offset - r, cy - r, r * 2, r * 2);

        using (SolidBrush bLeft = new SolidBrush(Blue))
        using (SolidBrush bRight = new SolidBrush(Amber))
        using (SolidBrush bLens = new SolidBrush(Cream))
        {
            g.FillEllipse(bLeft, left);
            g.FillEllipse(bRight, right);
            GraphicsState st = g.Save();
            using (GraphicsPath pLeft = new GraphicsPath())
            using (GraphicsPath pRight = new GraphicsPath())
            {
                pLeft.AddEllipse(left);
                pRight.AddEllipse(right);
                g.SetClip(pLeft);
                g.SetClip(pRight, CombineMode.Intersect);
                g.FillEllipse(bLens, right);
            }
            g.Restore(st);
        }
    }

    public static Icon CreateAppIcon()
    {
        Bitmap bmp = new Bitmap(32, 32);
        using (Graphics g = Graphics.FromImage(bmp))
        {
            g.Clear(Color.Transparent);
            DrawDiscs(g, 16f, 16f, 9.5f);
        }
        return Icon.FromHandle(bmp.GetHicon());
    }
}

static class InstallEngine
{
    public const string WeightsName = "model-sft3c.pt";

    [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    static extern IntPtr SendMessageTimeout(
        IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam,
        uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);

    public static string DefaultInstallDir()
    {
        string localApp = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(localApp, "Programs", "Alethic");
    }

    public static void AddToUserPath(string dir, Action<string> log)
    {
        string cur = Environment.GetEnvironmentVariable("Path", EnvironmentVariableTarget.User) ?? "";
        string want = dir.TrimEnd('\\').ToLowerInvariant();
        string[] parts = cur.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (string p in parts)
        {
            if (p.Trim().TrimEnd('\\').ToLowerInvariant() == want)
            {
                log("Already on User PATH: " + dir);
                return;
            }
        }
        string next = cur.TrimEnd(';') == "" ? dir : cur.TrimEnd(';') + ";" + dir;
        Environment.SetEnvironmentVariable("Path", next, EnvironmentVariableTarget.User);
        UIntPtr res;
        SendMessageTimeout((IntPtr)0xffff, 0x001A, UIntPtr.Zero, "Environment", 2, 2000, out res);
        log("Added to User PATH: " + dir);
    }

    public static void RegisterUninstaller(string installDir)
    {
        try
        {
            using (RegistryKey key = Registry.CurrentUser.CreateSubKey(
                @"Software\Microsoft\Windows\CurrentVersion\Uninstall\Alethic"))
            {
                if (key == null) return;
                key.SetValue("DisplayName", "Alethic Agent");
                key.SetValue("DisplayVersion", "0.1.0");
                key.SetValue("Publisher", "Alethic-300M");
                key.SetValue("InstallLocation", installDir);
                key.SetValue("UninstallString", "\"" + Path.Combine(installDir, "uninstall.bat") + "\"");
                key.SetValue("NoModify", 1, RegistryValueKind.DWord);
                key.SetValue("NoRepair", 1, RegistryValueKind.DWord);
            }
        }
        catch { }
    }

    static void StopRunningAlethic(string installDir, Action<string> log)
    {
        string targetNorm = Path.GetFullPath(installDir).TrimEnd('\\').ToLowerInvariant();
        string[] names = new[] { "alethic", "alethic-agent" };
        bool stoppedAny = false;
        foreach (string name in names)
        {
            Process[] procs;
            try { procs = Process.GetProcessesByName(name); }
            catch { continue; }
            foreach (Process p in procs)
            {
                try
                {
                    if (p.Id == Process.GetCurrentProcess().Id) continue;
                    bool match = true;
                    try
                    {
                        string exe = p.MainModule != null ? p.MainModule.FileName : null;
                        if (!string.IsNullOrEmpty(exe))
                        {
                            string dir = Path.GetDirectoryName(Path.GetFullPath(exe)).TrimEnd('\\').ToLowerInvariant();
                            match = (dir == targetNorm) || dir.EndsWith("\\programs\\alethic");
                        }
                    }
                    catch { }
                    if (match)
                    {
                        log("Closing running " + p.ProcessName + ".exe (PID " + p.Id + ")...");
                        stoppedAny = true;
                        try
                        {
                            ProcessStartInfo tk = new ProcessStartInfo("taskkill", "/F /T /PID " + p.Id);
                            tk.UseShellExecute = false;
                            tk.CreateNoWindow = true;
                            Process tkp = Process.Start(tk);
                            tkp.WaitForExit(4000);
                        }
                        catch { }
                        try { if (!p.HasExited) p.Kill(); } catch { }
                        try { p.WaitForExit(5000); } catch { }
                    }
                }
                catch { }
            }
        }
        if (stoppedAny) Thread.Sleep(700);
    }

    static void CopyFileRetry(string src, string dst)
    {
        for (int attempt = 0; attempt < 8; attempt++)
        {
            try
            {
                if (File.Exists(dst)) File.SetAttributes(dst, FileAttributes.Normal);
                File.Copy(src, dst, true);
                return;
            }
            catch (IOException) { if (attempt == 7) throw; }
            catch (UnauthorizedAccessException) { if (attempt == 7) throw; }
            Thread.Sleep(300);
        }
    }

    static void RemoveLegacyFiles(string installDir, Action<string> log)
    {
        string[] legacyFiles = new[] { "alethic-agent.exe", "alethic.cmd", "alethic-cli.cmd", "alethic.exe" };
        foreach (string name in legacyFiles)
        {
            string p = Path.Combine(installDir, name);
            if (!File.Exists(p)) continue;
            for (int attempt = 0; attempt < 6; attempt++)
            {
                try
                {
                    File.SetAttributes(p, FileAttributes.Normal);
                    File.Delete(p);
                    log("Replaced previous " + name);
                    break;
                }
                catch
                {
                    if (attempt < 5) Thread.Sleep(250);
                }
            }
        }
        string[] legacyDirs = new[] { "_internal", "venv" };
        foreach (string d in legacyDirs)
        {
            string p = Path.Combine(installDir, d);
            if (!Directory.Exists(p)) continue;
            for (int attempt = 0; attempt < 4; attempt++)
            {
                try { Directory.Delete(p, true); break; }
                catch { if (attempt < 3) Thread.Sleep(250); }
            }
        }
    }

    static int RunExe(string file, string args, Action<string> logLine)
    {
        ProcessStartInfo psi = new ProcessStartInfo(file, args);
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        Process p = Process.Start(psi);
        p.OutputDataReceived += (s, e) => { if (!string.IsNullOrEmpty(e.Data)) logLine(e.Data); };
        p.ErrorDataReceived += (s, e) => { if (!string.IsNullOrEmpty(e.Data)) logLine(e.Data); };
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        p.WaitForExit();
        return p.ExitCode;
    }

    static string FindExistingWeights(string installDir)
    {
        string target = Path.Combine(installDir, WeightsName);
        if (File.Exists(target)) return target;
        string legacyData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "alethic-agent", WeightsName);
        if (File.Exists(legacyData)) return legacyData;
        return null;
    }

    public static void Install(
        string installDir,
        bool installWeights,
        bool addToPath,
        Action<int, string> onProgress,
        Action<string> log,
        Func<bool> isCancelled)
    {
        string tempZip = Path.Combine(Path.GetTempPath(), "alethic-bundle-" + Guid.NewGuid().ToString("N") + ".zip");
        string tempStage = Path.Combine(Path.GetTempPath(), "alethic-stage-" + Guid.NewGuid().ToString("N"));
        string preservedWeights = null;
        string targetWeights = Path.Combine(installDir, WeightsName);

        try
        {
            onProgress(5, "Checking running Alethic processes...");
            StopRunningAlethic(installDir, log);
            if (isCancelled()) throw new OperationCanceledException();

            onProgress(12, "Extracting standalone runtime...");
            Assembly asm = Assembly.GetExecutingAssembly();
            using (Stream src = asm.GetManifestResourceStream("bundle.zip"))
            {
                if (src == null) throw new InvalidOperationException("Embedded bundle.zip resource is missing.");
                using (FileStream dst = File.Create(tempZip))
                    src.CopyTo(dst);
            }

            ZipFile.ExtractToDirectory(tempZip, tempStage);
            string nested = Path.Combine(tempStage, "alethic-agent");
            string srcDir = Directory.Exists(nested) ? nested : tempStage;
            if (isCancelled()) throw new OperationCanceledException();

            onProgress(32, "Replacing old Alethic harness files...");
            Directory.CreateDirectory(installDir);
            if (File.Exists(targetWeights))
            {
                preservedWeights = Path.Combine(Path.GetTempPath(), "alethic-weights-" + Guid.NewGuid().ToString("N") + ".pt");
                File.Move(targetWeights, preservedWeights);
                log("Preserved existing model weights.");
            }
            RemoveLegacyFiles(installDir, log);

            string[] allFiles = Directory.GetFiles(srcDir, "*", SearchOption.AllDirectories);
            foreach (string dir in Directory.GetDirectories(srcDir, "*", SearchOption.AllDirectories))
                Directory.CreateDirectory(dir.Replace(srcDir, installDir));
            for (int i = 0; i < allFiles.Length; i++)
            {
                if (isCancelled()) throw new OperationCanceledException();
                string file = allFiles[i];
                string destFile = file.Replace(srcDir, installDir);
                Directory.CreateDirectory(Path.GetDirectoryName(destFile));
                CopyFileRetry(file, destFile);
                if (i % 40 == 0 || i == allFiles.Length - 1)
                    onProgress(35 + (int)(30.0 * (i + 1) / Math.Max(1, allFiles.Length)), "Copying harness files (" + (i + 1) + "/" + allFiles.Length + ")...");
            }

            if (preservedWeights != null && File.Exists(preservedWeights) && !File.Exists(targetWeights))
                File.Move(preservedWeights, targetWeights);

            string exePath = Path.Combine(installDir, "alethic.exe");
            if (!File.Exists(exePath)) exePath = Path.Combine(installDir, "alethic-agent.exe");

            onProgress(70, "Verifying embedded Python & PyTorch...");
            int rc = RunExe(exePath, "--selftest", log);
            if (rc != 0) throw new InvalidOperationException("Embedded library check failed with exit code " + rc);
            if (isCancelled()) throw new OperationCanceledException();

            if (!File.Exists(targetWeights))
            {
                string existingOther = FindExistingWeights(installDir);
                string sideBySide = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, WeightsName);
                if (File.Exists(sideBySide))
                {
                    onProgress(78, "Copying bundled model weights (1.2 GB)...");
                    log("Copying local " + WeightsName + " to " + installDir + "...");
                    File.Copy(sideBySide, targetWeights, true);
                }
                else if (existingOther != null && File.Exists(existingOther))
                {
                    onProgress(78, "Moving existing model weights into install folder...");
                    File.Move(existingOther, targetWeights);
                }
                else if (installWeights)
                {
                    onProgress(78, "Downloading SFT-3c model weights (1.2 GB)...");
                    log("Downloading SFT-3c weights from Hugging Face...");
                    int dl = RunExe(exePath, "--download-weights", log);
                    if (dl != 0)
                        log("Warning: model weights download did not complete; alethic will retry on first run.");
                }
            }
            else
            {
                log("Model weights verified: " + targetWeights);
            }

            onProgress(95, "Updating PATH and Windows uninstaller...");
            if (addToPath) AddToUserPath(installDir, log);
            RegisterUninstaller(installDir);
            onProgress(100, "Installation complete.");
        }
        catch
        {
            if (preservedWeights != null && File.Exists(preservedWeights))
            {
                try { if (!File.Exists(targetWeights)) File.Move(preservedWeights, targetWeights); }
                catch { }
            }
            throw;
        }
        finally
        {
            try { if (File.Exists(tempZip)) File.Delete(tempZip); } catch { }
            try { if (Directory.Exists(tempStage)) Directory.Delete(tempStage, true); } catch { }
        }
    }
}

class WizardForm : Form
{
    Panel _header;
    Panel _pageWelcome;
    Panel _pageProgress;
    Panel _pageDone;

    TextBox _dirBox;
    Label _spaceLbl;
    CheckBox _chkWeights;
    CheckBox _chkPath;
    Button _btnInstall;
    Button _btnCancel;

    Label _progTitle;
    ProgressBar _progBar;
    TextBox _logBox;

    Label _doneSummary;
    CheckBox _chkLaunch;
    Button _btnFinish;

    string _installedDir;
    volatile bool _cancelled;

    public WizardForm(string initialDir)
    {
        Text = "Alethic Agent Setup";
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false;
        MinimizeBox = true;
        StartPosition = FormStartPosition.CenterScreen;
        ClientSize = new Size(580, 440);
        BackColor = Brand.Bg;
        ForeColor = Brand.Cream;
        Font = new Font("Segoe UI", 9.5f);
        try { Icon = Brand.CreateAppIcon(); } catch { }

        BuildHeader();
        BuildWelcomePage(initialDir);
        BuildProgressPage();
        BuildDonePage();
        ShowPage(_pageWelcome);
    }

    void BuildHeader()
    {
        _header = new Panel();
        _header.Bounds = new Rectangle(0, 0, 580, 78);
        _header.BackColor = Brand.Card;
        _header.Paint += (s, e) =>
        {
            Brand.DrawDiscs(e.Graphics, 42f, 39f, 16f);
            using (Pen p = new Pen(Brand.Border))
                e.Graphics.DrawLine(p, 0, _header.Height - 1, _header.Width, _header.Height - 1);
        };

        Label title = new Label();
        title.Text = "Alethic Agent";
        title.Font = new Font("Segoe UI", 15f, FontStyle.Bold);
        title.ForeColor = Brand.Cream;
        title.Location = new Point(78, 15);
        title.AutoSize = true;

        Label sub = new Label();
        sub.Text = "Standalone 300M Coding Agent  ·  Zero Python or PyTorch setup";
        sub.Font = new Font("Segoe UI", 9f);
        sub.ForeColor = Brand.Muted;
        sub.Location = new Point(79, 44);
        sub.AutoSize = true;

        _header.Controls.Add(title);
        _header.Controls.Add(sub);
        Controls.Add(_header);
    }

    Button MakeButton(string text, Point loc, Size size, bool primary)
    {
        Button b = new Button();
        b.Text = text;
        b.Location = loc;
        b.Size = size;
        b.FlatStyle = FlatStyle.Flat;
        b.Cursor = Cursors.Hand;
        b.Font = new Font("Segoe UI", 9.5f, primary ? FontStyle.Bold : FontStyle.Regular);
        if (primary)
        {
            b.BackColor = Brand.Amber;
            b.ForeColor = Brand.Bg;
            b.FlatAppearance.BorderSize = 0;
        }
        else
        {
            b.BackColor = Brand.Card;
            b.ForeColor = Brand.Cream;
            b.FlatAppearance.BorderColor = Brand.Border;
            b.FlatAppearance.BorderSize = 1;
        }
        return b;
    }

    void BuildWelcomePage(string initialDir)
    {
        _pageWelcome = new Panel();
        _pageWelcome.Bounds = new Rectangle(0, 78, 580, 362);

        Label intro = new Label();
        intro.Text = "This wizard installs the Alethic harness and SFT-3c model onto your drive,\n"
                   + "replaces any earlier alethic-agent version, and adds `alethic` to PATH.";
        intro.ForeColor = Brand.Cream;
        intro.Location = new Point(28, 20);
        intro.Size = new Size(524, 42);

        Label locLbl = new Label();
        locLbl.Text = "Destination folder";
        locLbl.ForeColor = Brand.Muted;
        locLbl.Font = new Font("Segoe UI", 9f, FontStyle.Bold);
        locLbl.Location = new Point(28, 76);
        locLbl.AutoSize = true;

        _dirBox = new TextBox();
        _dirBox.Text = initialDir;
        _dirBox.Location = new Point(28, 99);
        _dirBox.Size = new Size(428, 26);
        _dirBox.BackColor = Brand.InputBg;
        _dirBox.ForeColor = Brand.Cream;
        _dirBox.BorderStyle = BorderStyle.FixedSingle;
        _dirBox.TextChanged += (s, e) => UpdateSpaceInfo();

        Button btnBrowse = MakeButton("Browse...", new Point(464, 97), new Size(88, 28), false);
        btnBrowse.Click += (s, e) =>
        {
            using (FolderBrowserDialog dlg = new FolderBrowserDialog())
            {
                dlg.Description = "Choose where to install Alethic Agent";
                dlg.SelectedPath = Directory.Exists(_dirBox.Text) ? _dirBox.Text : InstallEngine.DefaultInstallDir();
                if (dlg.ShowDialog(this) == DialogResult.OK)
                {
                    string picked = dlg.SelectedPath.TrimEnd('\\');
                    _dirBox.Text = picked.EndsWith("\\Alethic", StringComparison.OrdinalIgnoreCase)
                        ? picked
                        : Path.Combine(picked, "Alethic");
                }
            }
        };

        _spaceLbl = new Label();
        _spaceLbl.ForeColor = Brand.Muted;
        _spaceLbl.Font = new Font("Segoe UI", 8.5f);
        _spaceLbl.Location = new Point(28, 130);
        _spaceLbl.Size = new Size(524, 20);

        Panel box = new Panel();
        box.Bounds = new Rectangle(28, 160, 524, 126);
        box.BackColor = Brand.Card;
        box.Paint += (s, e) =>
        {
            using (Pen p = new Pen(Brand.Border))
                e.Graphics.DrawRectangle(p, 0, 0, box.Width - 1, box.Height - 1);
        };

        Label compTitle = new Label();
        compTitle.Text = "Components";
        compTitle.ForeColor = Brand.Amber;
        compTitle.Font = new Font("Segoe UI", 9f, FontStyle.Bold);
        compTitle.Location = new Point(14, 12);
        compTitle.AutoSize = true;

        Label harnessInfo = new Label();
        harnessInfo.Text = "[Required]  Alethic Agent Harness (alethic.exe + embedded Python 3.12 & PyTorch)";
        harnessInfo.ForeColor = Brand.Cream;
        harnessInfo.Location = new Point(14, 38);
        harnessInfo.AutoSize = true;

        _chkWeights = new CheckBox();
        _chkWeights.Text = "Install SFT-3c 300M model weights (1.2 GB download if not present)";
        _chkWeights.Checked = true;
        _chkWeights.ForeColor = Brand.Cream;
        _chkWeights.Location = new Point(14, 65);
        _chkWeights.AutoSize = true;

        _chkPath = new CheckBox();
        _chkPath.Text = "Add install folder to User PATH (run `alethic` in any directory)";
        _chkPath.Checked = true;
        _chkPath.ForeColor = Brand.Cream;
        _chkPath.Location = new Point(14, 93);
        _chkPath.AutoSize = true;

        box.Controls.AddRange(new Control[] { compTitle, harnessInfo, _chkWeights, _chkPath });

        _btnInstall = MakeButton("Install", new Point(352, 308), new Size(104, 32), true);
        _btnInstall.Click += (s, e) => StartInstall();

        _btnCancel = MakeButton("Cancel", new Point(464, 308), new Size(88, 32), false);
        _btnCancel.Click += (s, e) => Close();

        _pageWelcome.Controls.AddRange(new Control[] { intro, locLbl, _dirBox, btnBrowse, _spaceLbl, box, _btnInstall, _btnCancel });
        Controls.Add(_pageWelcome);
        UpdateSpaceInfo();
    }

    void UpdateSpaceInfo()
    {
        try
        {
            string root = Path.GetPathRoot(Path.GetFullPath(_dirBox.Text.Trim()));
            DriveInfo di = new DriveInfo(root);
            double freeGb = di.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
            bool hasWeights = File.Exists(Path.Combine(_dirBox.Text.Trim(), InstallEngine.WeightsName));
            _spaceLbl.Text = string.Format("Drive {0} free space: {1:0.1} GB  (requires ~1.8 GB total{2})",
                root, freeGb, hasWeights ? "; existing weights will be kept" : "");
        }
        catch
        {
            _spaceLbl.Text = "";
        }
    }

    void BuildProgressPage()
    {
        _pageProgress = new Panel();
        _pageProgress.Bounds = new Rectangle(0, 78, 580, 362);

        _progTitle = new Label();
        _progTitle.Text = "Preparing installation...";
        _progTitle.Font = new Font("Segoe UI", 10.5f, FontStyle.Bold);
        _progTitle.ForeColor = Brand.Cream;
        _progTitle.Location = new Point(28, 24);
        _progTitle.Size = new Size(524, 24);

        _progBar = new ProgressBar();
        _progBar.Location = new Point(28, 56);
        _progBar.Size = new Size(524, 22);
        _progBar.Style = ProgressBarStyle.Continuous;
        _progBar.Minimum = 0;
        _progBar.Maximum = 100;

        _logBox = new TextBox();
        _logBox.Multiline = true;
        _logBox.ReadOnly = true;
        _logBox.ScrollBars = ScrollBars.Vertical;
        _logBox.Location = new Point(28, 94);
        _logBox.Size = new Size(524, 198);
        _logBox.BackColor = Brand.Card;
        _logBox.ForeColor = Brand.Muted;
        _logBox.BorderStyle = BorderStyle.FixedSingle;
        _logBox.Font = new Font("Consolas", 8.5f);

        Button cancelBtn = MakeButton("Cancel", new Point(464, 308), new Size(88, 32), false);
        cancelBtn.Click += (s, e) => { _cancelled = true; cancelBtn.Enabled = false; _progTitle.Text = "Cancelling..."; };

        _pageProgress.Controls.AddRange(new Control[] { _progTitle, _progBar, _logBox, cancelBtn });
        Controls.Add(_pageProgress);
    }

    void BuildDonePage()
    {
        _pageDone = new Panel();
        _pageDone.Bounds = new Rectangle(0, 78, 580, 362);

        Label headline = new Label();
        headline.Text = "Alethic Agent has been installed!";
        headline.Font = new Font("Segoe UI", 13f, FontStyle.Bold);
        headline.ForeColor = Brand.Green;
        headline.Location = new Point(28, 28);
        headline.AutoSize = true;

        _doneSummary = new Label();
        _doneSummary.ForeColor = Brand.Cream;
        _doneSummary.Location = new Point(28, 66);
        _doneSummary.Size = new Size(524, 120);

        _chkLaunch = new CheckBox();
        _chkLaunch.Text = "Launch Alethic now in a new terminal window";
        _chkLaunch.Checked = true;
        _chkLaunch.ForeColor = Brand.Cream;
        _chkLaunch.Location = new Point(28, 206);
        _chkLaunch.AutoSize = true;

        _btnFinish = MakeButton("Finish", new Point(448, 308), new Size(104, 32), true);
        _btnFinish.Click += (s, e) =>
        {
            if (_chkLaunch.Checked && !string.IsNullOrEmpty(_installedDir))
            {
                string exe = Path.Combine(_installedDir, "alethic.exe");
                if (File.Exists(exe))
                {
                    ProcessStartInfo psi = new ProcessStartInfo(exe);
                    psi.WorkingDirectory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                    Process.Start(psi);
                }
            }
            Close();
        };

        _pageDone.Controls.AddRange(new Control[] { headline, _doneSummary, _chkLaunch, _btnFinish });
        Controls.Add(_pageDone);
    }

    void ShowPage(Panel page)
    {
        _pageWelcome.Visible = (page == _pageWelcome);
        _pageProgress.Visible = (page == _pageProgress);
        _pageDone.Visible = (page == _pageDone);
    }

    void AppendLog(string line)
    {
        if (IsDisposed) return;
        BeginInvoke(new Action(() => { _logBox.AppendText(line + "\r\n"); }));
    }

    void StartInstall()
    {
        string target = _dirBox.Text.Trim();
        if (string.IsNullOrEmpty(target))
        {
            MessageBox.Show(this, "Please select an install folder.", "Alethic Setup", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }
        _installedDir = Path.GetFullPath(target);
        bool doWeights = _chkWeights.Checked;
        bool doPath = _chkPath.Checked;
        _cancelled = false;
        _logBox.Clear();
        ShowPage(_pageProgress);

        Thread worker = new Thread(() =>
        {
            try
            {
                InstallEngine.Install(
                    _installedDir,
                    doWeights,
                    doPath,
                    (pct, msg) =>
                    {
                        if (!IsDisposed)
                        {
                            BeginInvoke(new Action(() =>
                            {
                                _progBar.Value = Math.Max(0, Math.Min(100, pct));
                                _progTitle.Text = msg;
                            }));
                        }
                    },
                    AppendLog,
                    () => _cancelled);

                if (!IsDisposed)
                {
                    BeginInvoke(new Action(() =>
                    {
                        _doneSummary.Text =
                            "Install folder:\r\n  " + _installedDir + "\r\n\r\n" +
                            "Open any new Command Prompt (cmd) or PowerShell window and run:\r\n" +
                            "  alethic";
                        ShowPage(_pageDone);
                    }));
                }
            }
            catch (OperationCanceledException)
            {
                if (!IsDisposed) BeginInvoke(new Action(() => { Close(); }));
            }
            catch (Exception ex)
            {
                AppendLog("ERROR: " + ex);
                if (!IsDisposed)
                {
                    BeginInvoke(new Action(() =>
                    {
                        MessageBox.Show(this, "Installation failed:\n\n" + ex.Message, "Alethic Setup",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                        ShowPage(_pageWelcome);
                    }));
                }
            }
        });
        worker.IsBackground = true;
        worker.Start();
    }
}

class AlethicInstaller
{
    [DllImport("user32.dll")]
    static extern bool SetProcessDPIAware();

    [DllImport("kernel32.dll")]
    static extern bool AttachConsole(int dwProcessId);

    static string ParseArgValue(string arg, string prefix)
    {
        if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            return arg.Substring(prefix.Length).Trim().Trim('"');
        return null;
    }

    [STAThread]
    static int Main(string[] args)
    {
        try { SetProcessDPIAware(); } catch { }

        bool quiet = false;
        bool noWeights = false;
        string customDir = null;

        foreach (string a in args)
        {
            string l = a.ToLowerInvariant();
            if (l == "/quiet" || l == "--quiet" || l == "-q") quiet = true;
            else if (l == "/no-weights" || l == "--no-weights") noWeights = true;
            else
            {
                string val = ParseArgValue(a, "/dir=") ?? ParseArgValue(a, "--dir=");
                if (!string.IsNullOrEmpty(val)) customDir = val;
            }
        }

        string installDir = string.IsNullOrEmpty(customDir)
            ? InstallEngine.DefaultInstallDir()
            : Path.GetFullPath(customDir);

        if (quiet)
        {
            try { AttachConsole(-1); } catch { }
            string logPath = Path.Combine(Path.GetTempPath(), "alethic-setup.log");
            Action<string> writeLog = msg =>
            {
                try { File.AppendAllText(logPath, msg + Environment.NewLine); } catch { }
                try { Console.WriteLine(msg); } catch { }
            };
            try
            {
                if (File.Exists(logPath)) File.Delete(logPath);
                writeLog("Installing Alethic Agent to " + installDir + " ...");
                InstallEngine.Install(
                    installDir,
                    !noWeights,
                    true,
                    (pct, msg) => writeLog("[" + pct + "%] " + msg),
                    writeLog,
                    () => false);
                writeLog("Alethic Agent installed cleanly in " + installDir);
                return 0;
            }
            catch (Exception ex)
            {
                writeLog("Install failed: " + ex);
                return 1;
            }
        }

        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new WizardForm(installDir));
        return 0;
    }
}
