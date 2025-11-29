% v2改自v1，比v1多了保存所有文件到rangebin_rx文件夹下

clear; close all; clc;
%% ========================================================================
% 【零】 参数配置区：所有参数都可在此更改
% ------------------------------------------------------------------------
% --- 1. 文件路径与文件名配置 ---
pathname = 'D:\MSc\Dissertation\Data\250826\test9'; 
raw_bin_fname = 'test9_radar_Raw_0.bin';

% --- 1.1 输出文件夹和文件名配置 (基础路径) ---
base_output_dir_name = 'RangeFFT';
base_output_dir = fullfile(pathname, base_output_dir_name);

% 确保基础输出目录存在
if ~exist(base_output_dir, 'dir')
    mkdir(base_output_dir);
    fprintf('已创建基础输出目录: %s\n', base_output_dir);
end

% 数据文件名称 (路径将在运行时动态确定)
resampled_mat_fname = 'test9_radar_20hz_4.mat';
range_fft_mat_fname = 'test9_radar_RangeFFT.mat'; 
aligned_trimmed_mat_fname = 'test9_NeuLogRadar_aligned_trimmed.mat';

% NeuLog 文件仍在原路径下读取
neulog_csv_fname = 'test9_neulog.csv'; 

% 视频和图形文件名称
video_output_name = 'RangeFFTvideo.mp4';
cross_corr_fig_name = 'CrossCorr.fig'; 
spectrum_fig_name = 'Spectrum.fig'; 
filted_phase_fig_name = 'filted_phase_1_2.fig'; 
pulse_peak_fig_name = 'pulse_peak.fig'; 

cross_corr_delay_record = 14; % 已知的粗略时延，单位：秒

% --- 2. 雷达硬件参数 ---
Periodicity = 50.0e-3; % 单个物理帧的时长，单位：s
original_fs = 20; % 原始帧率
target_fs = 20;   % 目标帧率
fs = 5e6; % ADC Sampling frequency sps
c = 3e8;
da = 1.9e-3; 
nTX = 1; 
nRX = 4; 
FreqStart = 77e9;
Slope = 29.982e12; % Chirp rate Hz/s
IdleTime = 100e-6; % s
ADCStartTime = 6e-6; % s
ADC_Samples = 256; 
RampEndTime = 60e-6; % s
nChirpLoop = 1; 
nVX = nTX * nRX; % 虚拟天线数 (即接收天线数)

% --- 3. Range FFT 视频生成参数 ---
video_nRangeBin = 20; % 视频中显示的距离格数量
range_fft_limits = [20, 100]; % 视频振幅色阶范围 (dB)
video_start_time_s = 200; % 视频起始时间 (s)
video_duration_s = 20; % 视频持续时间 (s)

% --- 4. 单元格相位分析参数 ---
detrend_para = 4; % detrend函数的参数

% --- 5. 互相关分析参数 ---
cross_corr_delay_range = 2;   % 在粗略时延基础上，向前后扩展的搜索范围
trim_start_s = 90; % 裁剪掉对齐后的数据前多少秒，单位：秒
cross_corr_plot_duration_s = 200; % 互相关对齐波形对比图的持续时间

% --- 6. 最终绘图时间段参数 ---
final_plot_start_s = 100;
final_plot_duration_s = 100;

% 初始化分析索引 (将在阶段三被用户赋值)
analysis_range_bin_index = []; 
analysis_rx_index = [];
output_dir = []; % 最终输出路径 (将在阶段三被赋值)
%% ========================================================================
% 【一】 阶段一：原始数据重采样并生成中间文件
% ------------------------------------------------------------------------
fprintf('--- 阶段一：开始处理原始数据并重采样 ---\n');

% 计算相关参数
Expected_Num_SamplesPerFrame = ADC_Samples * nVX * nChirpLoop * 2;
fpath = fullfile(pathname, raw_bin_fname);

% 读取数据 (此部分不变)
fprintf('正在读取文件: %s\n', fpath);
fid = fopen(fpath, 'r');
if fid == -1
    error('无法打开文件，请检查文件路径是否正确。');
end
fseek(fid, 0, 'eof');
DataSize = ftell(fid);
fclose(fid);
nFrame = floor(DataSize / (2 * Expected_Num_SamplesPerFrame));
fprintf('文件总大小: %d 字节\n', DataSize);
fprintf('总帧数: %d\n', nFrame);

fid = fopen(fpath, 'r');
total_samples_to_read = nFrame * Expected_Num_SamplesPerFrame;
data = fread(fid, total_samples_to_read, 'int16');
fclose(fid);

% 数据的重构和维度调整: (ADC_Samples x nChirpLoop x nVX x nFrame)
data = reshape(data, 4, []);
pre_adc_data = zeros(2, size(data,2)); 
pre_adc_data(1, :) = data(1, :) + 1i * data(3, :); 
pre_adc_data(2, :) = data(2, :) + 1i * data(4, :); 

adc_data = reshape(pre_adc_data, ADC_Samples, nVX, nChirpLoop, nFrame);
adc_data = permute(adc_data, [1, 3, 2, 4]); 

fprintf('成功重构数据。原始数据的尺寸为 (ADC_Samples x nChirpLoop x nVX x nFrame)：\n');
disp(size(adc_data));

% 数据重采样
fprintf('开始将数据从 %dHz 重采样到 %dHz...\n', original_fs, target_fs);
squeezed_adc_data = squeeze(adc_data); 
resampled_adc_data = resample(squeezed_adc_data, target_fs, original_fs, 'Dimension', 3); 
n_frames_resampled = size(resampled_adc_data, 3);

fprintf('重采样完成。重采样后数据的尺寸为 (Range Samples x nVX x new_nFrame)：\n');
disp(size(resampled_adc_data));

% 阶段一数据保存 (临时保存在基础目录)
temp_fpath = fullfile(base_output_dir, resampled_mat_fname);
save(temp_fpath, 'resampled_adc_data', 'adc_data', 'original_fs', 'target_fs', ...
    'c', 'da', 'nTX', 'nRX', 'nVX', 'FreqStart', 'Slope', 'IdleTime', ...
    'ADCStartTime', 'ADC_Samples', 'fs', 'RampEndTime', 'nChirpLoop', ...
    'Periodicity', 'nFrame', '-v7.3');
fprintf('阶段一完成：数据和参数已成功保存至: %s\n', temp_fpath);

%% ========================================================================
% 【二】 阶段二：计算和保存 1D Range FFT 数据及相位解缠绕
% ------------------------------------------------------------------------
fprintf('\n--- 阶段二：开始计算并保存 Range FFT 数据 ---\n');
% 提取数据和关键参数 
n_frames = n_frames_resampled;
Nr_full = ADC_Samples;
fc = FreqStart + (ADCStartTime + ADC_Samples / 2 / fs) * Slope;
lambda = c / fc;
delta_R = c / (2 * Slope * (ADC_Samples / fs)); 
fprintf('雷达和处理参数定义完成。计算出的距离分辨率为 $%.4f$ m。\n', delta_R);

% 定义和计算截取范围
nRangeBinsToSave = video_nRangeBin + 1; 
if nRangeBinsToSave > Nr_full
    error('指定的距离格数量超出FFT点数。');
end
range_indices = 1:nRangeBinsToSave;
Nr = length(range_indices);
y_axis = (range_indices - 1)' * delta_R; % 距离轴 (米)

% 1D Range FFT 序列计算
fprintf('开始计算指定区域（距离格 0-%d）的 Range FFT...\n', Nr-1);
range_profile_full = fft(resampled_adc_data, Nr_full, 1);
range_fft_data = range_profile_full(range_indices, :, :); % (Nr x nVX x n_frames)
fprintf('指定区域 Range FFT 计算完成。\n');

% 相位解缠绕与最终数据提取
fprintf('开始进行1D相位解缠绕并提取振幅数据...\n');
unwrapped_phase_fft = zeros(Nr, nVX, n_frames);
for r = 1:Nr
    for v = 1:nVX 
        time_series_data = squeeze(range_fft_data(r, v, :));
        unwrapped_phase = unwrap(angle(time_series_data));
        unwrapped_phase_fft(r, v, :) = unwrapped_phase;
    end
end
magnitude_range_fft = abs(range_fft_data);
fprintf('相位解缠绕与数据提取完成。\n');

% 阶段二数据保存 (临时保存在基础目录)
temp_fpath_2 = fullfile(base_output_dir, range_fft_mat_fname);
save(temp_fpath_2, 'magnitude_range_fft', 'unwrapped_phase_fft', 'y_axis', ...
    'delta_R', 'fc', 'lambda', 'c', 'FreqStart', 'Slope', ...
    'ADC_Samples', 'fs', 'nVX', 'target_fs', '-v7.3');
fprintf('阶段二完成：Range FFT 数据已成功保存至: %s\n', temp_fpath_2);

%% ========================================================================
% 【三】 阶段三：生成 Range FFT 振幅对比视频、用户输入并确定最终保存路径
% ------------------------------------------------------------------------
fprintf('\n--- 阶段三：开始生成 Range FFT 振幅对比视频 ---\n');
% 定义时间范围
end_time_s = video_start_time_s + video_duration_s;
start_frame_idx = round(video_start_time_s * target_fs) + 1;
end_frame_idx = round(end_time_s * target_fs);
frames_to_process = start_frame_idx:end_frame_idx;
n_frames_to_process = length(frames_to_process);

% 检查帧范围是否有效
if start_frame_idx < 1 || end_frame_idx > size(magnitude_range_fft, 3)
    error('计算出的视频帧范围超出可用数据。请检查视频起始时间或数据总时长。');
end

% 提取视频所需的振幅数据段
video_mag_fft = magnitude_range_fft(:, :, frames_to_process);

fprintf('振幅色阶范围: $[%.2f, %.2f]$ dB\n', range_fft_limits(1), range_fft_limits(2));

% Range Bin 对应的距离值
range_bin_distances = y_axis; 

% --- 视频写入设置（先打开文件，稍后写入） ---
video_path = fullfile(base_output_dir, video_output_name);
video_writer = VideoWriter(video_path, 'MPEG-4');
video_writer.FrameRate = target_fs;
video_writer.Quality = 100;
open(video_writer);
fprintf('正在初始化视频写入，文件将暂时保存在基础目录: %s\n', video_path);
% ---------------------------------------------

figure('Position', [100, 100, 1600, 400], 'Name', 'Range FFT 振幅对比视频');
fprintf('正在生成视频...\n');

for i = 1:n_frames_to_process
    current_time_s = (frames_to_process(i) - 1) / target_fs;
    sgtitle(['Range FFT 振幅对比 (时间: ', num2str(current_time_s, '%.2f'), 's)'], 'FontWeight', 'bold');
    
    for v = 1:nVX
        subplot(1, nVX, v); 
        current_range_profile_db = 20 * log10(video_mag_fft(:, v, i));
        
        plot(range_bin_distances, current_range_profile_db, 'LineWidth', 2);
        hold on;
        
        % 绘制 Range Bin 索引线
        for r = 1:Nr
            line([range_bin_distances(r), range_bin_distances(r)], range_fft_limits, ...
                 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'HandleVisibility', 'off');
        end
        hold off;
        
        grid on;
        box on;
        ylim(range_fft_limits); 
        xlim([range_bin_distances(1), range_bin_distances(end)]);
        
        % 设置横坐标标签为每个 Range Bin 对应的距离值
        xticks(range_bin_distances);
        xticklabels(cellstr(num2str(range_bin_distances, '%.2f'))); 
        xtickangle(45); 
        
        title(['RX ', num2str(v), ' Range FFT (Bin 0 to ', num2str(video_nRangeBin), ')']);
        xlabel('距离 (米)');
        if v == 1
            ylabel('幅度 (dB)');
        end
    end
    
    % 保存当前帧到视频文件
    frame = getframe(gcf);
    writeVideo(video_writer, frame);
    drawnow;
    
end
close(video_writer); % 关闭视频写入
close(gcf); % 关闭视频窗口
fprintf('视频生成完成。\n');

% --- 用户输入，确定分析单元格 ---
fprintf('\n请根据视频，确定要分析的距离格（Range Bin）和 RX 天线。\n');
fprintf('Range FFT 的距离格索引从 1 开始。\n');
analysis_range_bin_index = input('请输入要分析的距离格索引（Range Bin Index, 例如：7）： ');
if isempty(analysis_range_bin_index)
    error('未输入距离格索引。脚本已终止。');
end

prompt_rx = sprintf('请输入要分析的 RX 天线索引（RX Index, 1 到 %d）： ', nVX);
analysis_rx_index = input(prompt_rx); 

if isempty(analysis_rx_index) || analysis_rx_index < 1 || analysis_rx_index > nVX
    error('RX 索引输入无效。脚本已终止。');
end

fprintf('您已选择距离格索引：%d，RX 索引：%d\n', ...
    analysis_range_bin_index, analysis_rx_index);

% --- 动态创建最终输出文件夹并移动文件 ---
specific_output_dir_name = ['RangeBin', num2str(analysis_range_bin_index), '_RX', num2str(analysis_rx_index)];
output_dir = fullfile(base_output_dir, specific_output_dir_name);

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('已创建嵌套输出目录: %s\n', output_dir);
end

% 移动 Stage 1 & 2 的数据文件到最终目录
movefile(temp_fpath, fullfile(output_dir, resampled_mat_fname));
movefile(temp_fpath_2, fullfile(output_dir, range_fft_mat_fname));
fprintf('数据文件已移动到最终目录。\n');

% 移动视频文件到最终目录
movefile(video_path, fullfile(output_dir, video_output_name));
fprintf('视频文件已移动到最终目录： %s\n', fullfile(output_dir, video_output_name));


%% ========================================================================
% 【四】 阶段四：互相关分析与数据裁剪
% ------------------------------------------------------------------------
fprintf('\n--- 阶段四：开始互相关分析并裁剪数据 ---\n');

% 加载 NeuLog 数据
neulog_fpath = fullfile(pathname, neulog_csv_fname);
fprintf('正在加载 NeuLog 数据: %s\n', neulog_fpath);
try
    fileID = fopen(neulog_fpath, 'r');
    if fileID == -1
        error('无法打开NeuLog文件。请检查文件路径是否正确。');
    end
    data = textscan(fileID, '%q%f%f', 'Delimiter', ';', 'headerlines', 7, 'EmptyValue', NaN);
    fclose(fileID);
    timeStrings = data{1};
    pulseData = data{2};
    respirationData = data{3};
    numDataPoints = length(timeStrings);
    totalSeconds = zeros(numDataPoints, 1);
    invalidRows = false(numDataPoints, 1);
    for i = 1:numDataPoints
        timeStr = strrep(timeStrings{i}, '''', '');
        timeParts = strsplit(timeStr, ':');
        if length(timeParts) >= 3
            totalSeconds(i) = str2double(timeParts{1}) * 3600 + ...
                                         str2double(timeParts{2}) * 60 + ...
                                         str2double(timeParts{3});
        else
            invalidRows(i) = true;
        end
    end
    respirationData(invalidRows) = [];
    pulseData(invalidRows) = [];
    totalSeconds(invalidRows) = [];
    fprintf('NeuLog 数据加载成功。\n');
catch ME
    error('无法加载NeuLog数据文件: %s\n', ME.message);
end

% 提取指定单元格的相位时间序列 (使用用户选择的参数)
fprintf('提取指定单元格 (Range Bin: %d, RX: %d) 的相位时间序列...\n', ...
    analysis_range_bin_index, analysis_rx_index);
% 注意：unwrapped_phase_fft 在阶段二已计算并仍在内存中

radar_phase_data = squeeze(unwrapped_phase_fft(analysis_range_bin_index, analysis_rx_index, :));

% --- 对雷达相位数据进行带通滤波 (0.1Hz - 2.5Hz) 仅用于互相关分析 ---
fprintf('正在对雷达相位数据进行带通滤波 (0.1Hz - 2.5Hz)，此结果仅用于互相关...\n');
filter_freqs = [0.1, 2.5]; % 单位: Hz
fs_radar = target_fs;
[b, a] = butter(4, filter_freqs * 2 / fs_radar, 'bandpass');
filtered_radar_phase = filtfilt(b, a, radar_phase_data);
fprintf('雷达数据滤波完成，此结果用于互相关分析。\n');

% 数据预处理
fprintf('对雷达和 NeuLog 数据进行去趋势和去均值处理...\n');
radar_detrended = filtered_radar_phase; 
neulog_processed = respirationData - mean(respirationData);
neulog_pulse_processed = pulseData - mean(pulseData);
fprintf('数据处理完成。\n');

% 计算互相关与寻找最佳延迟 (已修复归一化问题)
fprintf('开始计算互相关并寻找最佳延迟...\n');
search_start = cross_corr_delay_record - cross_corr_delay_range;
search_end = cross_corr_delay_record + cross_corr_delay_range;
[C_temp, lags] = xcorr(radar_detrended, neulog_processed);
N_radar = length(radar_detrended);
N_neulog = length(neulog_processed);
K = length(lags);
C = zeros(1, K);

% 互相关归一化 (根据重叠长度)
for i = 1:K
    n = lags(i);
    if n >= 0
        idx_radar = n+1 : N_radar;
        idx_neulog = 1 : N_neulog - n;
    else
        idx_radar = 1 : N_radar + n;
        idx_neulog = -n+1 : N_neulog;
    end
    
    if isempty(idx_radar) || isempty(idx_neulog)
        C(i) = 0;
        continue;
    end
    
    energy_radar_overlap = sum(abs(radar_detrended(idx_radar)).^2);
    energy_neulog_overlap = sum(abs(neulog_processed(idx_neulog)).^2);
    normalization_factor = sqrt(energy_radar_overlap * energy_neulog_overlap);
    
    if normalization_factor > 0
        C(i) = C_temp(i) / normalization_factor;
    else
        C(i) = 0;
    end
end
lags_in_seconds = lags / target_fs;
search_indices = find(lags_in_seconds >= search_start & lags_in_seconds <= search_end);

if isempty(search_indices)
    error('在指定的延迟范围（%.2f s到%.2f s）内没有找到数据点。', search_start, search_end);
end

[~, max_idx_in_search] = max(abs(C(search_indices)));
overall_max_idx = search_indices(max_idx_in_search);
estimated_delay = lags_in_seconds(overall_max_idx);
max_corr = C(overall_max_idx);
fprintf('在 $%.2f$ s 到 $%.2f$ s 的范围内，找到最大互相关峰值。\n', search_start, search_end);
fprintf('估测的延迟为: $%.4f$ 秒\n', estimated_delay);
fprintf('互相关计算完成。\n');

% 绘制互相关结果和对齐波形图
fprintf('正在绘制互相关分析图表（CrossCorr）...\n');
figure('Name', '互相关分析结果', 'NumberTitle', 'off');
subplot(2, 1, 1);
plot(lags_in_seconds, C, 'k-');
hold on;
plot(estimated_delay, max_corr, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
text(estimated_delay, max_corr, sprintf(' %.2f s', estimated_delay), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10);
hold off;
title('雷达与NeuLog数据的互相关结果');
xlabel('延迟 (秒)');
ylabel('归一化相关值');
grid on;
box on;
ylim([-1, 1]);
xlim([search_start-2 search_end+2]);

subplot(2, 1, 2);
hold on;
title('去均值数据对齐后波形对比');
xlabel('时间 (秒)');
grid on;
box on;
yyaxis left;
neulog_plot_indices_3 = 1:min(length(neulog_processed), round(cross_corr_plot_duration_s * target_fs));
t_neulog_plot_3 = (0:(length(neulog_plot_indices_3)-1)) / target_fs;
plot(t_neulog_plot_3, neulog_processed(neulog_plot_indices_3), 'b-', 'DisplayName', 'NeuLog 呼吸');
ylabel('NeuLog 呼吸振幅 (Arb)');

yyaxis right;
radar_start_offset_s = estimated_delay;
radar_end_offset_s = cross_corr_plot_duration_s + estimated_delay;
radar_start_idx_3 = round(radar_start_offset_s * target_fs) + 1;
radar_end_idx_3 = round(radar_end_offset_s * target_fs);
radar_start_idx_3 = max(1, radar_start_idx_3);
radar_end_idx_3 = min(length(radar_detrended), radar_end_idx_3);
radar_plot_indices_3 = radar_start_idx_3:radar_end_idx_3;
t_radar_plot_3 = (0:(length(radar_plot_indices_3)-1)) / target_fs;
plot(t_radar_plot_3, radar_detrended(radar_plot_indices_3), 'r-', 'DisplayName', '雷达相位');
ylabel('雷达相位振幅 (Arb)');
legend('Location', 'best');
hold off;
fprintf('互相关分析图表绘制完成。\n');

% 保存互相关图
savefig(gcf, fullfile(output_dir, cross_corr_fig_name));
fprintf('互相关图已保存至: %s\n', fullfile(output_dir, cross_corr_fig_name));

% 数据对齐与裁剪
fprintf('\n开始对齐并裁剪数据（裁剪前%d秒）...\n', trim_start_s);
trimmed_length_samples = length(neulog_processed) - round(trim_start_s * target_fs);
if trimmed_length_samples <= 0
    error('裁剪时长过长，导致数据长度非正。请调整 trim_start_s。');
end

neulog_trim_start_idx = round(trim_start_s * target_fs) + 1;
radar_align_trim_start_idx = round((estimated_delay + trim_start_s) * target_fs) + 1;
radar_align_trim_end_idx = radar_align_trim_start_idx + trimmed_length_samples - 1;

if radar_align_trim_end_idx > size(unwrapped_phase_fft, 3)
    radar_align_trim_end_idx = size(unwrapped_phase_fft, 3);
end
trimmed_length_samples = radar_align_trim_end_idx - radar_align_trim_start_idx + 1;
neulog_trim_end_idx = neulog_trim_start_idx + trimmed_length_samples - 1;
if neulog_trim_end_idx > length(neulog_processed)
    neulog_trim_end_idx = length(neulog_processed);
end

% 裁剪 NeuLog 数据
neulog_respiration_data_trimmed = neulog_processed(neulog_trim_start_idx:neulog_trim_end_idx);
neulog_pulse_data_trimmed = neulog_pulse_processed(neulog_trim_start_idx:neulog_trim_end_idx);

% 裁剪雷达数据 (所有 Range Bin, 所有 RX)
magnitude_range_fft_trimmed = magnitude_range_fft(:, :, radar_align_trim_start_idx:radar_align_trim_end_idx);
trimmed_unfiltered_phase_fft = unwrapped_phase_fft(:, :, radar_align_trim_start_idx:radar_align_trim_end_idx);

% **新增：设计并应用1-2Hz滤波器**
fprintf('\n**开始对整个相位 FFT 矩阵进行1-2Hz滤波...**\n');
% 滤波参数定义 (1-2Hz)
filter_freqs_1_2hz = [1, 2];
[b_1_2, a_1_2] = butter(4, filter_freqs_1_2hz * 2 / target_fs, 'bandpass');
% 创建一个新矩阵来存储1-2Hz滤波结果
filtered_1_2hz_phase_fft = zeros(size(trimmed_unfiltered_phase_fft));
n_ranges = size(trimmed_unfiltered_phase_fft, 1);
n_vx = size(trimmed_unfiltered_phase_fft, 2);

% 遍历每个单元格（Range Bin x RX）并应用新的滤波器
for r = 1:n_ranges
    for v = 1:n_vx
        time_series = squeeze(trimmed_unfiltered_phase_fft(r, v, :));
        filtered_series = filtfilt(b_1_2, a_1_2, time_series);
        filtered_1_2hz_phase_fft(r, v, :) = filtered_series;
    end
end
fprintf('整个相位 FFT 矩阵 1-2Hz 滤波完成。\n');

% ========================================================================
fprintf('\n--- 脉搏数据峰值检测和掩码生成 ---\n');
% ------------------------------------------------------------------------
% 1. 对 NeuLog 脉搏数据进行 1-2Hz 带通滤波
% ------------------------------------------------------------------------
[b_pulse, a_pulse] = butter(4, [1 2]/(target_fs/2), 'bandpass');
filtered_pulse_data = filtfilt(b_pulse, a_pulse, neulog_pulse_data_trimmed);
fprintf('NeuLog脉搏数据已完成1-2Hz带通滤波。\n');

% ------------------------------------------------------------------------
% 2. 基于滤波后的数据进行峰值检测
% ------------------------------------------------------------------------
min_peak_distance_samples = round(0.5 * target_fs);
min_peak_prominence = 0.2;
[~, locs_filtered] = findpeaks(filtered_pulse_data, 'MinPeakDistance', min_peak_distance_samples, 'MinPeakProminence', min_peak_prominence);
filtered_peak_mask = zeros(size(filtered_pulse_data));
filtered_peak_mask(locs_filtered) = 1;
fprintf('基于滤波数据的峰值检测已完成。\n');

% ------------------------------------------------------------------------
% 3. 基于未滤波的数据进行峰值检测
% ------------------------------------------------------------------------
[~, locs_unfiltered] = findpeaks(neulog_pulse_data_trimmed, 'MinPeakDistance', min_peak_distance_samples, 'MinPeakProminence', min_peak_prominence);
unfiltered_peak_mask = zeros(size(neulog_pulse_data_trimmed));
unfiltered_peak_mask(locs_unfiltered) = 1;
fprintf('基于未滤波数据的峰值检测已完成。\n');

% ------------------------------------------------------------------------
% 4. 创建扩展峰值掩码
% ------------------------------------------------------------------------
fprintf('正在创建扩展峰值掩码...\n');
filtered_peak_mask_wide = filtered_peak_mask;
unfiltered_peak_mask_wide = unfiltered_peak_mask;
% 找到所有峰值索引
idx_filtered = find(filtered_peak_mask == 1);
idx_unfiltered = find(unfiltered_peak_mask == 1);
% 扩展滤波掩码 (向前向后各扩展一帧)
for i = 1:length(idx_filtered)
    peak_idx = idx_filtered(i);
    if peak_idx > 1
        filtered_peak_mask_wide(peak_idx - 1) = 1;
    end
    if peak_idx < length(filtered_peak_mask)
        filtered_peak_mask_wide(peak_idx + 1) = 1;
    end
end
% 扩展未滤波掩码
for i = 1:length(idx_unfiltered)
    peak_idx = idx_unfiltered(i);
    if peak_idx > 1
        unfiltered_peak_mask_wide(peak_idx - 1) = 1;
    end
    if peak_idx < length(unfiltered_peak_mask)
        unfiltered_peak_mask_wide(peak_idx + 1) = 1;
    end
end
fprintf('扩展峰值掩码创建完成。\n');

% ------------------------------------------------------------------------
% 5. 保存所有数据到 .mat 文件
% ------------------------------------------------------------------------
final_data_path = fullfile(output_dir, aligned_trimmed_mat_fname);
save(final_data_path, ...
    'neulog_respiration_data_trimmed', ...
    'neulog_pulse_data_trimmed', ...
    'filtered_pulse_data', ...
    'unfiltered_peak_mask', ...
    'filtered_peak_mask', ...
    'unfiltered_peak_mask_wide', ... 
    'filtered_peak_mask_wide', ... 
    'filtered_1_2hz_phase_fft', ... 
    'trimmed_unfiltered_phase_fft', ... 
    'magnitude_range_fft_trimmed', ...
    'estimated_delay', 'trim_start_s', 'target_fs', ...
    'analysis_range_bin_index', 'analysis_rx_index', ... 
    '-v7.3');
fprintf('数据保存成功！保存至: %s\n', final_data_path);

%% ========================================================================
% 【五】 阶段五：绘制指定单元格的相位图和 NeuLog 频率谱图
% ------------------------------------------------------------------------
fprintf('\n--- 阶段五：开始绘制指定单元格的相位分析图和 NeuLog 频率谱图 ---\n');

% 从对齐和裁剪后的数据中提取最终分析的时间段
final_plot_end_s = final_plot_start_s + final_plot_duration_s;
fprintf('将从对齐数据中分析 $%.2f$s 到 $%.2f$s 的时间段。\n', final_plot_start_s, final_plot_end_s);

% 计算时间段在裁剪后数据中的索引
start_sample_idx = round(final_plot_start_s * target_fs) + 1;
end_sample_idx = round(final_plot_end_s * target_fs);
indices_to_plot = start_sample_idx:end_sample_idx;
n_samples_to_plot = length(indices_to_plot);

% 检查索引是否有效
if isempty(indices_to_plot) || indices_to_plot(1) < 1 || indices_to_plot(end) > length(neulog_respiration_data_trimmed)
    error('指定的最终绘图时间段超出已裁剪数据的范围。请检查 final_plot_start_s 和 final_plot_duration_s。');
end

% 1. 提取雷达相位数据并进行去趋势 (使用用户选择的 Range Bin 和 RX)
fprintf('正在提取雷达相位数据...\n');
radar_phase_data_segment = squeeze(trimmed_unfiltered_phase_fft(analysis_range_bin_index, analysis_rx_index, indices_to_plot));
t = (0:n_samples_to_plot - 1) / target_fs;
detrended_phase = detrend(radar_phase_data_segment, detrend_para);
fprintf('雷达相位数据提取和去趋势完成。\n');

% 2. 对雷达数据进行 FFT
fprintf('正在计算雷达相位频率谱...\n');
N = length(detrended_phase);
Fs = target_fs;
frequency_axis_hz = (0:N-1) * Fs / N;
frequency_axis_cpm = frequency_axis_hz * 60;
magnitude_spectrum_db = 20 * log10(abs(fft(detrended_phase, N)));
fprintf('雷达频率谱计算完成。\n');

% 3. 提取 NeuLog 数据并进行频率谱分析
fprintf('正在计算 NeuLog 频率谱...\n');
pulse_data_segment = neulog_pulse_data_trimmed(indices_to_plot);
respiration_data_segment = neulog_respiration_data_trimmed(indices_to_plot);
L = length(pulse_data_segment);
Y_pulse = fft(pulse_data_segment);
Y_resp = fft(respiration_data_segment);

% 计算单边谱 (脉搏)
P2_pulse = abs(Y_pulse/L);
P1_pulse = P2_pulse(1:L/2+1);
P1_pulse(2:end-1) = 2*P1_pulse(2:end-1);
% 计算单边谱 (呼吸)
P2_resp = abs(Y_resp/L);
P1_resp = P2_resp(1:L/2+1);
P1_resp(2:end-1) = 2*P1_resp(2:end-1);

f = Fs*(0:(L/2))/L;
freq_bpm = f * 60;
P1_pulse_dB = 20*log10(P1_pulse);
P1_resp_dB = 20*log10(P1_resp);
fprintf('NeuLog 频率谱计算完成。\n');

% 4. 绘制四子图 (Spectrum)
fprintf('开始绘制最终分析四子图（Spectrum）...\n');
figure('Position', [100, 100, 1200, 900]);
sgtitle(sprintf('雷达相位 (Range Bin: %d, RX: %d) 与 NeuLog 频率谱分析', analysis_range_bin_index, analysis_rx_index));

% 左上: 雷达相位时间图
subplot(2, 2, 1);
plot(t, detrended_phase, 'LineWidth', 1.5);
grid on;
title(['去趋势后的相位时间图']);
xlabel('时间 (秒)');
ylabel('相位 (弧度)');

% 左下: 雷达相位频率谱
subplot(2, 2, 3);
valid_idx = find(frequency_axis_cpm >= 0 & frequency_axis_cpm <= 120);
plot(frequency_axis_cpm(valid_idx), magnitude_spectrum_db(valid_idx), 'LineWidth', 1.5);
grid on;
title('雷达相位频率谱');
xlabel('频率 (次数/分钟)');
ylabel('幅度 (dB)');
max_y = max(magnitude_spectrum_db(valid_idx));
if isfinite(max_y)
    ylim([max(0, max_y - 30), max_y * 1.1]);
else
    ylim([-60, 20]); 
end
xlim([0 120]);

% 右上: 脉搏频率谱
subplot(2, 2, 2);
plot(freq_bpm, P1_pulse_dB, 'b-', 'LineWidth', 2);
title(sprintf('脉搏频率谱 (%ds-%ds)', final_plot_start_s, final_plot_end_s));
xlabel('频率 (次/分钟)');
ylabel('幅度 (dB)');
grid on;
box on;
xlim([40 120]);
ylim([0 inf]);

% 右下: 呼吸频率谱
subplot(2, 2, 4);
plot(freq_bpm, P1_resp_dB, 'g-', 'LineWidth', 2);
title(sprintf('呼吸频率谱 (%ds-%ds)', final_plot_start_s, final_plot_end_s));
xlabel('频率 (次/分钟)');
ylabel('幅度 (dB)');
grid on;
box on;
xlim([0 60]);
ylim([0 inf]);

% 保存四子图
savefig(gcf, fullfile(output_dir, spectrum_fig_name));
fprintf('四子图（Spectrum）已保存至: %s\n', fullfile(output_dir, spectrum_fig_name));

% --- 绘制 1-2Hz 滤波后雷达相位图 (filted_phase_1_2) ---
fprintf('\n--- 绘制：1-2Hz 滤波后雷达相位波形与频率谱 ---\n');
figure('Name', '雷达相位（1-2Hz滤波）分析', 'NumberTitle', 'off');

% 提取数据段
radar_phase_filtered_segment = squeeze(filtered_1_2hz_phase_fft(analysis_range_bin_index, analysis_rx_index, indices_to_plot));

% 上子图: 1-2Hz滤波后波形
subplot(2, 1, 1);
plot(t, radar_phase_filtered_segment, 'LineWidth', 1.5);
grid on;
title(sprintf('1-2Hz 滤波后雷达相位波形 (Range Bin: %d, RX: %d)', analysis_range_bin_index, analysis_rx_index));
xlabel('时间 (秒)');
ylabel('相位 (弧度)');

% 下子图: 1-2Hz滤波后频谱
subplot(2, 1, 2);
N_filtered = length(radar_phase_filtered_segment);
Fs_filtered = target_fs;
frequency_axis_hz_filtered = (0:N_filtered-1) * Fs_filtered / N_filtered;
frequency_axis_cpm_filtered = frequency_axis_hz_filtered * 60;
magnitude_spectrum_db_filtered = 20 * log10(abs(fft(radar_phase_filtered_segment, N_filtered)));
valid_idx_filtered = find(frequency_axis_cpm_filtered >= 0 & frequency_axis_cpm_filtered <= 120);
plot(frequency_axis_cpm_filtered(valid_idx_filtered), magnitude_spectrum_db_filtered(valid_idx_filtered), 'LineWidth', 1.5);
grid on;
title('1-2Hz 滤波后雷达相位频率谱');
xlabel('频率 (次数/分钟)');
ylabel('幅度 (dB)');
max_y_filtered = max(magnitude_spectrum_db_filtered(valid_idx_filtered));
if isfinite(max_y_filtered)
    ylim([max_y_filtered - 20, max_y_filtered + 10]);
else
    ylim([-100, -60]);
end
xlim([0 120]);
fprintf('1-2Hz 滤波后雷达相位图表绘制完成。\n');

% 保存 1-2Hz 滤波相位图
savefig(gcf, fullfile(output_dir, filted_phase_fig_name));
fprintf('1-2Hz 滤波相位图已保存至: %s\n', fullfile(output_dir, filted_phase_fig_name));

% --- 绘制脉搏波形与峰值对比 (pulse_peak) ---
figure('Name', 'NeuLog 脉搏波形与峰值检测双图对比', 'NumberTitle', 'off');
fprintf('正在绘制 %.2fs 到 %.2fs 的脉搏数据和峰值（pulse_peak）...\n', final_plot_start_s, final_plot_end_s);

% 提取绘图所需的数据段
plot_start_idx = round(final_plot_start_s * target_fs) + 1;
plot_end_idx = round(final_plot_end_s * target_fs);
plot_indices = plot_start_idx:plot_end_idx;
t_plot = (0:length(plot_indices)-1) / target_fs;

% ------------------------------------------------------------------------
% 绘制 subplot 1: 滤波数据波形 + 滤波峰值
% ------------------------------------------------------------------------
subplot(2,1,1);
pulse_data_to_plot_filtered = filtered_pulse_data(plot_indices);
% 找到并标记该数据段中的滤波峰值
locs_in_plot_indices_filtered = find(filtered_peak_mask(plot_indices) == 1);
peaks_to_plot_filtered = pulse_data_to_plot_filtered(locs_in_plot_indices_filtered);
t_peaks_filtered = t_plot(locs_in_plot_indices_filtered);
% 绘制波形和峰值
plot(t_plot, pulse_data_to_plot_filtered, 'b-', 'LineWidth', 1.5);
hold on;
plot(t_peaks_filtered, peaks_to_plot_filtered, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
hold off;
grid on;
box on;
title(sprintf('基于滤波数据的脉搏波形与峰值检测 (%ds-%ds)', final_plot_start_s, final_plot_end_s));
xlabel('时间 (秒)');
ylabel('滤波后的幅度');
legend('滤波后数据', '检测到的峰值');

% ------------------------------------------------------------------------
% 绘制 subplot 2: 未滤波数据波形 + 未滤波峰值
% ------------------------------------------------------------------------
subplot(2,1,2);
pulse_data_to_plot_unfiltered = neulog_pulse_data_trimmed(plot_indices);
% 找到并标记该数据段中的未滤波峰值
locs_in_plot_indices_unfiltered = find(unfiltered_peak_mask(plot_indices) == 1);
peaks_to_plot_unfiltered = pulse_data_to_plot_unfiltered(locs_in_plot_indices_unfiltered);
t_peaks_unfiltered = t_plot(locs_in_plot_indices_unfiltered);
% 绘制波形和峰值
plot(t_plot, pulse_data_to_plot_unfiltered, 'b-', 'LineWidth', 1.5);
hold on;
plot(t_peaks_unfiltered, peaks_to_plot_unfiltered, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
hold off;
grid on;
box on;
title(sprintf('基于未滤波数据的脉搏波形与峰值检测 (%ds-%ds)', final_plot_start_s, final_plot_end_s));
xlabel('时间 (秒)');
ylabel('原始幅度');
legend('原始数据', '检测到的峰值');

sgtitle('NeuLog 脉搏波形与峰值检测双图对比', 'FontWeight', 'bold');
fprintf('对比图表绘制完成。\n');

% 保存脉搏峰值图
savefig(gcf, fullfile(output_dir, pulse_peak_fig_name));
fprintf('脉搏峰值图已保存至: %s\n', fullfile(output_dir, pulse_peak_fig_name));

fprintf('\n--- 所有任务已成功完成！---\n');