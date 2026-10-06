% =========================================================================
% DEMO_VAD_6ch.m
% 
% 6-kanal kayit scripti — 25 tekrar VAD-uyumlu protokol
% 
% Donanim: ESP32-S3 + 2 EMG (AC-NB084) + 4 piezo (buffered)
% Sampling: 500 Hz, 12-bit ADC, USB serial @ 921600 baud
% 
% Protokol:
%   - 20s baseline (sessizlik) — VAD threshold kalibrasyonu icin
%   - 10 kelime x 25 tekrar = 250 kelime
%     - Her tekrar: 2s talimat goster + 5s kayit penceresi + 1s ara
%   - Aralarda iki kisa mola (3 dk)
%   - 20s tail (gurultu tabani)
% 
% Toplam sure: ~40 dk
% 
% Cikti:
%   recordings/session_NNN_YYYY-MM-DD_HH-MM/
%     ├── raw_6ch.csv
%     ├── markers.csv
%     └── session_meta.json
% =========================================================================

clear; clc; close all;

%% ====== AYARLAR ======
SERIAL_PORT     = "COM5";       % Windows'ta COM4, Mac/Linux'ta /dev/tty.usbserial-...
BAUD            = 921600;
FS              = 500;          % Hz
N_CHANNELS      = 6;

% Kelime listesi
WORDS = {'EVET', 'HAYIR', 'MERHABA', 'TESEKKURLER', 'SU', ...
         'YARDIM', 'TAMAM', 'DUR', 'GEL', 'GUNAYDIN'};

% Protokol parametreleri
N_REPS              = 25;       % kelime basina tekrar sayisi
BASELINE_SEC        = 20;       % bas sessizlik
TAIL_SEC            = 20;       % son gurultu tabani
INSTRUCTION_SEC     = 2;        % "EVET soyle" ekrani
WINDOW_SEC          = 5;        % konusma penceresi
INTER_REP_SEC       = 1;        % tekrarlar arasi
BREAK_AFTER_REPS    = [9, 17];  % 9. ve 17. tekrardan sonra mola
BREAK_DURATION_SEC  = 180;      % 3 dakika mola

% Operator / denek bilgisi
SUBJECT_ID  = "S01";
OPERATOR    = "Serhat";
NOTES       = "";

% Kelime sirasi randomize edilsin mi (her tekrar icinde 10 kelime karistirilir)
RANDOMIZE_WORDS = true;
RNG_SEED        = uint32(posixtime(datetime('now')));  % gunluk seed

%% ====== HAZIRLIK ======
rng(RNG_SEED);

% Oturum klasoru
sessionTime = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm');
sessionFolder = fullfile('recordings', ...
    sprintf('session_%s_%s', SUBJECT_ID, char(sessionTime)));
if ~exist(sessionFolder, 'dir')
    mkdir(sessionFolder);
end
fprintf('Oturum klasoru: %s\n', sessionFolder);

rawFilePath     = fullfile(sessionFolder, 'raw_6ch.csv');
markersFilePath = fullfile(sessionFolder, 'markers.csv');
metaFilePath    = fullfile(sessionFolder, 'session_meta.json');

% CSV header'lari
rawFid = fopen(rawFilePath, 'w');
fprintf(rawFid, 'timestamp_us,emg1,emg2,pzt1,pzt2,pzt3,pzt4\n');

markerFid = fopen(markersFilePath, 'w');
fprintf(markerFid, 'timestamp_us,event,word,repetition\n');

%% ====== SERIAL BAGLANTISI ======
fprintf('ESP32''ye baglaniliyor (%s @ %d)...\n', SERIAL_PORT, BAUD);
sp = serialport(SERIAL_PORT, BAUD);
configureTerminator(sp, "LF");
sp.Timeout = 5;

% Bagliantiyi temizle
flush(sp);
pause(1);

% PING test
writeline(sp, "PING");
pause(0.5);
flushed = false;
while sp.NumBytesAvailable > 0
    line = readline(sp);
    if contains(line, "PONG")
        fprintf('  ESP32 hazir.\n');
        flushed = true;
        break;
    end
end
if ~flushed
    error('ESP32 cevap vermiyor. Port ve baud rate''i kontrol et.');
end

%% ====== PRE-FLIGHT CHECKLIST ======
fprintf('\n========================================\n');
fprintf('  PRE-FLIGHT CHECKLIST\n');
fprintf('========================================\n');
checklistItems = {
    'Cilt hazirligi (alkol, jel) yapildi mi?'
    'Referans elektrot klavikulada mi?'
    'EMG1 (sol submental) takili mi?'
    'EMG2 (sag submental) takili mi?'
    'PZT1 (orta hat) tam temas?'
    'PZT2 (sol submandibular) tam temas?'
    'PZT3 (sag submandibular) tam temas?'
    'PZT4 (sol lateral boyun) tam temas?'
    'Kablolar arkaya alindi, gerginlik yok mu?'
    'ESP32 batarya/USB ile besleniyor mu?'
};
for i = 1:length(checklistItems)
    response = input(sprintf('[ ] %s (Enter ile devam): ', checklistItems{i}), 's');
end
fprintf('\nChecklist tamam. Kayit baslatiliyor...\n\n');

%% ====== KAYIT BASLAT ======
% Arka planda serial okuma icin callback kur
configureCallback(sp, "terminator", @(src, ~) onSerialData(src, rawFid, markerFid));

% START komutu
writeline(sp, "START");
sessionStartTic = tic;
pause(0.2);

%% ====== BASELINE ======
fprintf('========================================\n');
fprintf('  BASELINE: %ds sessizlik\n', BASELINE_SEC);
fprintf('========================================\n');
fprintf('Lutfen %d saniye boyunca SESSIZ kal, hareket etme.\n', BASELINE_SEC);
writeMarker(sp, 'baseline_start', '', 0);
countdownSeconds(BASELINE_SEC);
writeMarker(sp, 'baseline_end', '', 0);
fprintf('Baseline tamamlandi.\n\n');

%% ====== ANA PROTOKOL ======
totalReps = N_REPS;
breakSet  = BREAK_AFTER_REPS;

% Kelime sirasini ana donguden once kelime icinde randomize edecegiz
% Her tekrar bloku icinde 10 kelime gosterilecek (kelime sirasi karistirilarak)

for rep = 1:totalReps
    % Mola kontrolu
    if any(rep - 1 == breakSet) && rep > 1
        fprintf('\n========================================\n');
        fprintf('  MOLA: %d dakika\n', BREAK_DURATION_SEC/60);
        fprintf('========================================\n');
        fprintf('Su iç, çene-boyun gevşet. Elektrotlara dokunma!\n');
        writeMarker(sp, 'break_start', '', rep);
        countdownSeconds(BREAK_DURATION_SEC);
        writeMarker(sp, 'break_end', '', rep);
        
        % Mola sonrasi mini baseline
        fprintf('Mini baseline (10s sessizlik)...\n');
        writeMarker(sp, 'mini_baseline_start', '', rep);
        countdownSeconds(10);
        writeMarker(sp, 'mini_baseline_end', '', rep);
        fprintf('Devam ediliyor.\n\n');
    end
    
    fprintf('========== TEKRAR %d / %d ==========\n', rep, totalReps);
    
    % Kelime sirasini bu tekrar icin karistir
    if RANDOMIZE_WORDS
        wordOrder = randperm(length(WORDS));
    else
        wordOrder = 1:length(WORDS);
    end
    
    for wi = 1:length(wordOrder)
        word = WORDS{wordOrder(wi)};
        
        % --- Talimat ---
        clc;
        fprintf('Tekrar %d/%d - Kelime %d/10\n\n', rep, totalReps, wi);
        fprintf('==========================\n');
        fprintf('   HAZIRLAN:  %s\n', word);
        fprintf('==========================\n');
        writeMarker(sp, 'instruction', word, rep);
        pause(INSTRUCTION_SEC);
        
        % --- Konusma penceresi ---
        clc;
        fprintf('Tekrar %d/%d - Kelime %d/10\n\n', rep, totalReps, wi);
        fprintf('==========================\n');
        fprintf('  >>>  %s  <<<\n', word);
        fprintf('==========================\n');
        fprintf('(simdi soyle)\n');
        writeMarker(sp, 'window_start', word, rep);
        pause(WINDOW_SEC);
        writeMarker(sp, 'window_end', word, rep);
        
        % --- Ara ---
        pause(INTER_REP_SEC);
    end
end

%% ====== TAIL ======
fprintf('\n========================================\n');
fprintf('  TAIL: %ds gurultu tabani\n', TAIL_SEC);
fprintf('========================================\n');
fprintf('Son %d saniye sessiz kal.\n', TAIL_SEC);
writeMarker(sp, 'tail_start', '', 0);
countdownSeconds(TAIL_SEC);
writeMarker(sp, 'tail_end', '', 0);

%% ====== KAYIT DURDUR ======
writeline(sp, "STOP");
pause(0.5);
configureCallback(sp, "off");

% Kalan veriyi oku
while sp.NumBytesAvailable > 0
    line = readline(sp);
    parseAndWriteSerialLine(line, rawFid, markerFid);
end

fclose(rawFid);
fclose(markerFid);
clear sp;

elapsedSec = toc(sessionStartTic);
fprintf('\n========================================\n');
fprintf('  KAYIT TAMAMLANDI\n');
fprintf('========================================\n');
fprintf('Toplam sure: %.1f dakika\n', elapsedSec/60);
fprintf('Kelime: %d, Tekrar: %d (toplam %d kelime)\n', ...
    length(WORDS), N_REPS, length(WORDS)*N_REPS);

%% ====== METADATA ======
extraNotes = input('Oturuma dair notlar (yoksa Enter): ', 's');

meta = struct();
meta.session_id           = char(sessionTime);
meta.subject_id           = SUBJECT_ID;
meta.operator             = OPERATOR;
meta.date_time_iso        = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
meta.firmware_version     = "v1.0";
meta.script_version       = "DEMO_VAD_6ch v1.0";
meta.sampling_rate_hz     = FS;
meta.adc_bits             = 12;
meta.n_channels           = N_CHANNELS;
meta.channels             = {'emg1_left_submental', 'emg2_right_submental', ...
                             'pzt1_midline', 'pzt2_left_submandibular', ...
                             'pzt3_right_submandibular', 'pzt4_left_neck'};
meta.words                = WORDS;
meta.n_repetitions        = N_REPS;
meta.baseline_sec         = BASELINE_SEC;
meta.tail_sec             = TAIL_SEC;
meta.window_sec           = WINDOW_SEC;
meta.breaks_after_reps    = BREAK_AFTER_REPS;
meta.break_duration_sec   = BREAK_DURATION_SEC;
meta.randomize_words      = RANDOMIZE_WORDS;
meta.rng_seed             = double(RNG_SEED);
meta.duration_minutes     = elapsedSec/60;
meta.notes                = extraNotes;
meta.data_quality         = "unknown";  % oturum sonrasi Mert tarafindan guncellenir

% JSON yaz
jsonStr = jsonencode(meta, 'PrettyPrint', true);
fid = fopen(metaFilePath, 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('\nMetadata: %s\n', metaFilePath);
fprintf('Ham veri:  %s\n', rawFilePath);
fprintf('Markerlar: %s\n', markersFilePath);
fprintf('\nMert''e gonderilebilir.\n');

% =========================================================================
% YARDIMCI FONKSIYONLAR
% =========================================================================

function countdownSeconds(seconds)
    % Geriye sayim, ekrana yaz
    for s = seconds:-1:1
        fprintf('  %d... ', s);
        if mod(seconds - s + 1, 10) == 0, fprintf('\n'); end
        pause(1);
    end
    fprintf('\n');
end

function writeMarker(sp, eventName, word, rep)
    % ESP32'ye marker yolla — ESP32 timestamp'i ile birlikte CSV'ye yazacak
    msg = sprintf('MARK:%s|%s|%d', eventName, word, rep);
    writeline(sp, msg);
end

function onSerialData(src, rawFid, markerFid)
    % Serial callback — gelen her satiri uygun dosyaya yaz
    while src.NumBytesAvailable > 0
        try
            line = readline(src);
            parseAndWriteSerialLine(line, rawFid, markerFid);
        catch
            break;
        end
    end
end

function parseAndWriteSerialLine(line, rawFid, markerFid)
    line = strtrim(char(line));
    if isempty(line), return; end
    
    if startsWith(line, '#')
        % Yorum satiri veya marker
        if startsWith(line, '# MARKER,')
            % Format: # MARKER,<timestamp_us>,<event>|<word>|<rep>
            payload = extractAfter(line, '# MARKER,');
            parts = split(payload, ',');
            if length(parts) >= 2
                ts_us = parts{1};
                eventStr = strjoin(parts(2:end), ',');  % virgul iceriyorsa
                fields = split(eventStr, '|');
                if length(fields) >= 3
                    fprintf(markerFid, '%s,%s,%s,%s\n', ts_us, fields{1}, fields{2}, fields{3});
                else
                    fprintf(markerFid, '%s,%s,,\n', ts_us, eventStr);
                end
            end
        end
        return;
    end
    
    % Veri satiri: timestamp_us,ch0,...,ch5
    fprintf(rawFid, '%s\n', line);
end
