import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

plt.rcParams['font.family'] = 'IPAGothic'  # Windowsの場合MS Gothic

def LoadAudioData(filepath):
	# 音声データを読み込む
	mic, sr = sf.read(filepath)
	return mic, sr

def ComputeSfft(audiodata):
	# SFTを行う
	n_fft = 1024
	hop_length = 256
	return librosa.stft(audiodata, n_fft=n_fft, hop_length=hop_length)

def CreateFilter(filter_type, order ,fs , cutoff):
	# フィルタを作成する(IRR)
	nyq = fs / 2
	norm_cutoff = np.array(cutoff) / nyq
	b, a = butter(order, norm_cutoff, btype=filter_type, analog=False)
	return b, a

def PlotFilterResponse(b, a, fs, name):
	# フィルタの図示を行う
    w, h = freqz(b, a, worN=2048)
    freqs = w * fs / (2 * np.pi)
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, 20 * np.log10(abs(h)), label="Gain (dB)")
    plt.title(f'フィルタ特性: {name}')
    plt.xlabel('周波数 [Hz]')
    plt.ylabel('Gain [dB]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{name}_フィルタ特性.png")
    plt.close()

def PlotFFT(signal, fs, title, filename):
	# FFTを図示
    N = len(signal)
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    mag = np.abs(fft)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, 20 * np.log10(mag + 1e-6))
    plt.title(title)
    plt.xlabel('周波数 [Hz]')
    plt.ylabel('振幅 [dB]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{filename}")
    plt.close()


def RunFilter(audiodata, filtertype, filterorder, cutoff, fs):
	# フィルタを適用する
	b, a = CreateFilter(filtertype, filterorder, cutoff, fs)
	filtered = filtfilt(b, a, audiodata)
	return filtered

def ApplyFilters(order_seq, cutoff_seq, type_seq, audiodata, fs, refdata):
    # フィルタを適用する
	changedata = audiodata.copy()
	try:
		for i in range(3):
			ftype = type_seq[i]
			order = order_seq[i]
			cutoff = cutoff_seq[i]
			b, a = CreateFilter(ftype, order, fs, cutoff)
			changedata = filtfilt(b, a, changedata)
		score = GetScorePoint(changedata, refdata)
		return score, (type_seq, order_seq, cutoff_seq), changedata
	except Exception as e:
		return -np.inf, None, None

def SutdyFilter(filterlist, audiodata, refdata, fs):
	# 渡されたAudioDataを読み込み，重ね掛けで順繰りにフィルタをかけていく
	# 重ね掛けるフィルタのオーダ数，カットオフ周波数をstep10で変更し，GetScorePointのスコアの一番高かった設定値を保存する
	# ここはjoblibでマルチスレッド化を行うようにする
	# 探索対象の次数とカットオフ周波数
    orders = [2, 4, 6]
    cutoffs = {
        'low': np.arange(100, 5000, 500),
        'high': np.arange(1000, 8000, 500),
        'bandpass': [(low, high) for low in range(100, 3000, 500) for high in range(low+1000, 8000, 1000)]
    }

    best_score = -np.inf
    best_config = None
    best_output = None

    results = []

    for type_seq in filterlist:
        def generate_params():
            for o1 in orders:
                for o2 in orders:
                    for o3 in orders:
                        for c1 in cutoffs[type_seq[0]]:
                            for c2 in cutoffs[type_seq[1]]:
                                for c3 in cutoffs[type_seq[2]]:
                                    yield [o1, o2, o3], [c1, c2, c3]

        tasks = (ApplyFilters(order_seq, cutoff_seq, type_seq, audiodata, fs, refdata)
                 for order_seq, cutoff_seq in generate_params())

        # マルチスレッドで評価
        result = Parallel(n_jobs=-1, prefer="threads")(delayed(lambda x: x) for x in tasks)

        for score, config, output in result:
            if score > best_score:
                best_score = score
                best_config = config
                best_output = output

    print("=== 最良の構成 ===")
    print("Filter types: ", best_config[0])
    print("Orders: ", best_config[1])
    print("Cutoffs: ", best_config[2])
    print("Score: ", best_score)

    return best_output, best_config, best_score

def GetScorePoint(filteraudiodata, answerdata):
	# ここは点数を出す処理
	resultscore = 0
	return resultscore


def main():
	# メイン処理を行う
	# まずは、音声データを読み込む
	MicData1Path = f"./mic1.wav"
	MicData2Path = f"./mic2.wav"
	MicData3Path = f"./mic3.wav"

	RefData1Path = f"./ref1.wav"
	RefData2Path = f"./ref2.wav"
	RefData3Path = f"./ref3.wav"

	SampleBDDataPath = f"./sampleBD.wav"
	SampleHHDataPath = f"./sampleHH.wav"
	SampleSDDataPath = f"./sampleSD.wav"

	# 音声データを読み込み
	MicData1, MicData1_sr = LoadAudioData(MicData1Path)
	MicData2, MicData2_sr = LoadAudioData(MicData2Path)
	MicData3, MicData3_sr = LoadAudioData(MicData3Path)

	RefData1, RefData1_sr = LoadAudioData(RefData1Path)
	RefData2, RefData2_sr = LoadAudioData(RefData2Path)
	RefData3, RefData3_sr = LoadAudioData(RefData3Path)

	# SFFT変換を行う
	MicData1_fft = ComputeSfft(MicData1)
	MicData2_fft = ComputeSfft(MicData2)
	MicData3_fft = ComputeSfft(MicData3)

	RefData1_fft = ComputeSfft(RefData1)
	RefData2_fft = ComputeSfft(RefData2)
	RefData3_fft = ComputeSfft(RefData3)

	AudioDataList = [[MicData1, RefData1], [MicData2, RefData2], [MicData3, RefData3]]

	FilterList=[["low", "high", "bandpass"],["high", "low", "bandpass"],["high", "bandpass", "low"],["bandpass", "high", "low"],["bandpass", "low", "high"],["low", "bandpass", "high"]]
	for idx, (mic, ref) in enumerate(AudioDataList):
		fs = MicData1_sr
		PlotFFT(mic, fs, f"mic{idx+1} - フィルタ前FFT", f"fft_before_{idx+1}.png")
		best_out, best_cfg, best_score = SutdyFilter(FilterList, mic, ref, fs)
		sf.write(f"best_filtered_{idx+1}.wav", best_out, fs)
		PlotFFT(best_out, fs, f"mic{idx+1} - フィルタ後FFT", f"fft_after_{idx+1}.png")
		print(best_cfg)
		print(best_score)


if __name__ == '__main__':
    main()