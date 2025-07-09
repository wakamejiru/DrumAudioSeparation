import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
from itertools import permutations

# evaluation.py から GetScorePoint, alignment, bss_eval_sources をインポート
# GetScorePoint は、ref_data_np, mic_data_np, est_data_np (全てNumPy配列) を受け取るように
# evaluation.py 側で修正されている前提です。
from evaluation import GetScorePoint, alignment, bss_eval_sources

# matplotlibの日本語表示設定 (Windowsの場合)
plt.rcParams['font.family'] = 'MS Gothic' # Windowsの場合
# Mac/Linuxの場合は 'IPAGothic' など、システムに合ったフォントを設定してください
# 例: plt.rcParams['font.family'] = 'IPAGothic'
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP'] # こちらも試す価値あり
# plt.rcParams['axes.unicode_minus'] = False # マイナス記号を正しく表示

def LoadAudioData(filepaths):
    """
    指定された複数の音声ファイルパスから音声データを読み込み、結合して返します。
    すべてのファイルは同じサンプリングレートを持つと仮定します。
    
    Args:
        filepaths (list): 音声ファイルのパスのリスト。
        
    Returns:
        tuple: (combined_data, sr)
               - combined_data (np.ndarray): 結合された音声データ (サンプル数 x チャンネル数)
               - sr (int): サンプリング周波数
    """
    all_data = []
    sr = 0
    for path in filepaths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")
        data, current_sr = sf.read(path)
        if sr == 0:
            sr = current_sr
        elif sr != current_sr:
            raise ValueError("全てのファイルのサンプリングレートが一致しません。")
        all_data.append(data)
    
    # データ長を揃える（最短の長さに合わせる）
    min_len = min(len(d) for d in all_data)
    # NumPy配列を結合し、(サンプル数 x チャンネル数) に転置
    combined_data = np.array([d[:min_len] for d in all_data]).T
    
    return combined_data, sr

def ComputeSfft(audiodata, n_fft=1024, hop_length=256):
    """
    STFT (Short-Time Fourier Transform) を行います。
    
    Args:
        audiodata (np.ndarray): 音声データ (サンプル数 x チャンネル数)。
        n_fft (int): FFTの窓長。
        hop_length (int): ホップ長。
        
    Returns:
        np.ndarray: STFT結果。多チャンネルの場合は librosa が自動で処理。
    """
    # librosa.stft はモノラル入力 (1D配列) を想定しているため、
    # 多チャンネルの場合は各チャンネルごとにSTFTを実行し、結合する必要があります。
    # ここでは、SutdyFilter内で直接STFT結果は使わないため、最初のチャンネルのみSTFTを計算する例を示します。
    if audiodata.ndim > 1:
        return librosa.stft(audiodata[:, 0], n_fft=n_fft, hop_length=hop_length)
    else:
        return librosa.stft(audiodata, n_fft=n_fft, hop_length=hop_length)

def CreateFilter(filter_type, order ,fs , cutoff):
    # フィルタを作成する(IRR)
    nyq = fs / 2
    
    # カットオフ周波数を正規化
    # isinstance のチェックに tuple を追加
    if isinstance(cutoff, (list, np.ndarray, tuple)): # ★ ここを修正
        norm_cutoff = [c / nyq for c in cutoff]
    else:
        norm_cutoff = cutoff / nyq
        
    b, a = butter(order, norm_cutoff, btype=filter_type, analog=False)
    return b, a

def PlotFilterResponse(b, a, fs, name):
    """
    フィルタの周波数応答を図示し、PNGファイルとして保存します。
    """
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
    """
    信号のFFT（振幅スペクトル）を図示し、PNGファイルとして保存します。
    多チャンネル信号の場合は、最初のチャンネルのみをプロットします。
    """
    if signal.ndim > 1:
        signal = signal[:, 0] # 最初のチャンネルのみを使用

    N = len(signal)
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    mag = np.abs(fft)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, 20 * np.log10(mag + 1e-6)) # 0dB表示を避けるために小さな値を加える
    plt.title(title)
    plt.xlabel('周波数 [Hz]')
    plt.ylabel('振幅 [dB]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{filename}")
    plt.close()

def ApplyFilters(order_seq, cutoff_seq, type_seq, audiodata, fs, refdata_np):
    """
    フィルタのシーケンスをオーディオデータに適用し、そのSDRスコアを計算します。
    この関数は joblib で並列処理されることを意図しています。
    
    Args:
        order_seq (list): 各フィルタの次数を格納したリスト (例: [4, 4, 4])
        cutoff_seq (list): 各フィルタのカットオフ周波数。バンドパス/バンドストップの場合は [low, high]
        type_seq (list): 各フィルタの種類を格納したリスト (例: ['lowpass', 'highpass', 'bandpass'])
        audiodata (np.ndarray): フィルタを適用する元のオーディオデータ (サンプル数 x チャンネル数)
        fs (float): オーディオデータのサンプリング周波数
        refdata_np (np.ndarray): GetScorePoint関数に渡す参照データ (サンプル数 x チャンネル数)
        
    Returns:
        tuple: (score_tuple, (type_seq, order_seq, cutoff_seq), changedata)
               - score_tuple (tuple): GetScorePointから返されるSDRスコア (obsSDR, estSDR, impSDR)
               - (type_seq, order_seq, cutoff_seq) (tuple): 適用されたフィルタのパラメータ
               - changedata (np.ndarray): フィルタ適用後のオーディオデータ
               または、エラー発生時は ((-np.inf, チャンネル数分のゼロ配列, チャンネル数分のゼロ配列), None, None)
    """
    changedata = audiodata.copy()
    num_channels = audiodata.shape[1]

    try:
        # type_seqの長さに合わせてループ
        for i in range(len(type_seq)):
            ftype = type_seq[i]
            order = order_seq[i]
            cutoff = cutoff_seq[i]
            
            # CreateFilterの引数順序に注意: (filter_type, order, fs, cutoff)
            b, a = CreateFilter(ftype, order, fs, cutoff) 
            
            # filtfilt は多チャンネルの場合、axis=0 でサンプル軸に沿ってフィルタリング
            changedata = filtfilt(b, a, changedata, axis=0)
        
        # GetScorePoint を、NumPy配列を受け取るように修正した場合の呼び出し
        # refdata_np: 参照データ NumPy配列 (samples x channels)
        # audiodata: フィルタ適用前の元データ (mic_data_np に相当)
        # changedata: フィルタ適用後のデータ (est_data_np に相当)
        score_obsSDR, score_estSDR, score_impSDR = GetScorePoint(refdata_np, audiodata, changedata)
        
        # ApplyFilters の戻り値 `score` は GetScorePoint の返り値 (obsSDR, estSDR, impSDR) の全体を指すと解釈
        score_tuple = (score_obsSDR, score_estSDR, score_impSDR) 
        
        return score_tuple, (type_seq, order_seq, cutoff_seq), changedata
    except Exception as e:
        # エラー発生時のデバッグ情報
        print(f"ApplyFiltersでエラーが発生しました: {e}")
        # SDRの結果もチャンネル数分の配列を想定し、-infまたはゼロ配列で返す
        return (np.full(num_channels, -np.inf), np.zeros(num_channels), np.zeros(num_channels)), None, None

def SutdyFilter(filterlist_permutations, audiodata_combined, refdata_combined, fs):
    """
    渡されたオーディオデータと参照データに対して、指定されたフィルタの組み合わせと
    パラメータのすべての可能性を試行し、最も高いSDR改善量を持つ設定を探索します。
    joblib を用いてマルチスレッドで処理を並列化します。
    
    Args:
        filterlist_permutations (list): フィルタタイプの順列のリスト (例: [['lowpass', 'highpass', 'bandpass'], ...])
        audiodata_combined (np.ndarray): フィルタを適用する元の多チャンネルオーディオデータ (サンプル数 x チャンネル数)
        refdata_combined (np.ndarray): スコア計算のための多チャンネル参照データ (サンプル数 x チャンネル数)
        fs (float): サンプリング周波数
        
    Returns:
        tuple: (best_output, best_config, best_score_full_tuple)
               - best_output (np.ndarray): 最もSDR改善量の高かったフィルタ適用後のオーディオデータ
               - best_config (tuple): 最もSDR改善量の高かったフィルタ設定 (type_seq, order_seq, cutoff_seq)
               - best_score_full_tuple (tuple): 最もSDR改善量の高かった (obsSDR, estSDR, impSDR) のタプル
    """
    # 探索対象の次数とカットオフ周波数
    orders = [2, 4, 6]
    cutoffs = {
        'lowpass': np.arange(100, 5000, 500).tolist(), # キーを 'lowpass' に修正
        'highpass': np.arange(1000, 8000, 500).tolist(), # キーを 'highpass' に修正
        # バンドパスの範囲はタプルのリストにする
        'bandpass': [(low, high) for low in np.arange(100, 3000, 500) for high in np.arange(low + 1000, 8000, 1000) if low < high]
    }

    best_cumulative_impSDR = -np.inf # SDR改善量の合計値で最適化
    best_config = None
    best_output = None
    best_score_full_tuple = None # GetScorePointからの完全なスコアタプルを保存

    all_tasks = []

    # フィルタタイプのすべての順列を試す
    for type_seq in filterlist_permutations:
        # 各フィルタのタイプに対応するカットオフ周波数のリストを取得
        cutoffs_for_f1 = cutoffs[type_seq[0]]
        cutoffs_for_f2 = cutoffs[type_seq[1]]
        cutoffs_for_f3 = cutoffs[type_seq[2]]

        for o1 in orders:
            for o2 in orders:
                for o3 in orders:
                    for c1 in cutoffs_for_f1:
                        for c2 in cutoffs_for_f2:
                            for c3 in cutoffs_for_f3:
                                # delayed 関数に渡す引数は、直接呼び出す関数の引数と一致させる
                                all_tasks.append(delayed(ApplyFilters)(
                                    [o1, o2, o3], [c1, c2, c3], type_seq,
                                    audiodata_combined, fs, refdata_combined
                                ))
        
    print(f"合計 {len(all_tasks)} のフィルタ設定を試行します...")
    # マルチスレッドで評価を実行
    # tqdm を使用して進捗表示を追加すると便利です: from tqdm import tqdm
    # results_all = Parallel(n_jobs=-1, prefer="threads")(tqdm(all_tasks))
    results_all = Parallel(n_jobs=-1, prefer="threads")(all_tasks) # ★修正済み

    for score_tuple, config, output in results_all:
        # エラーでNoneが返された場合や、スコアが有効でない場合はスキップ
        if config is not None and score_tuple[2] is not None and not np.isinf(score_tuple[2]).any() and not np.isnan(score_tuple[2]).any():
            # score_tuple[2] は impSDR の配列
            current_impSDR_sum = np.sum(score_tuple[2]) # SDR改善量の合計値で比較
            
            if current_impSDR_sum > best_cumulative_impSDR:
                best_cumulative_impSDR = current_impSDR_sum
                best_config = config
                best_output = output
                best_score_full_tuple = score_tuple # 完全なスコアタプルを保存

    print("\n=== 最良の構成 ===")
    if best_config:
        print(f"Filter types: {best_config[0]}")
        print(f"Orders: {best_config[1]}")
        print(f"Cutoffs: {best_config[2]}")
        print(f"Best cumulative SDR Improvement (sum of impSDR across channels): {best_cumulative_impSDR:.4f} [dB]")
        if best_score_full_tuple:
            print(f"Observed SDR (before filter): {best_score_full_tuple[0]} [dB]")
            print(f"Estimated SDR (after filter): {best_score_full_tuple[1]} [dB]")
            print(f"SDR Improvement per channel: {best_score_full_tuple[2]} [dB]")
    else:
        print("最適なフィルタ構成は見つかりませんでした。")

    # 最終的な戻り値
    return best_output, best_config, best_score_full_tuple

def main():
    # メイン処理を行う
    print("--- フィルタ最適化処理開始 ---")

    # ここから、ファイル名が規則的でないことを想定したファイルパスの個別指定
    # この部分の設計意図を維持します。
    MicData1Path = "./mic1.wav"
    MicData2Path = "./mic2.wav"
    MicData3Path = "./mic3.wav"

    RefData1Path = "./ref1.wav"
    RefData2Path = "./ref2.wav"
    RefData3Path = "./ref3.wav"

    # 以下は、今回の処理では直接使用しませんが、元のコードにあったため残しておきます。
    # SampleBDDataPath = "./sampleBD.wav"
    # SampleHHDataPath = "./sampleHH.wav"
    # SampleSDDataPath = "./sampleSD.wav"

    # LoadAudioData 関数がファイルパスのリストを受け取るように設計されているため、
    # 個別に指定したパスをリストにまとめて渡します。
    mic_filepaths_list = [MicData1Path, MicData2Path, MicData3Path]
    ref_filepaths_list = [RefData1Path, RefData2Path, RefData3Path]

    # 音声データを読み込み
    try:
        # LoadAudioData は複数のファイルパスを受け取り、(サンプル数 x チャンネル数) のNumPy配列を返す
        MicData_combined, fs = LoadAudioData(mic_filepaths_list)
        RefData_combined, _ = LoadAudioData(ref_filepaths_list) # fsはMicData_combinedから取得済みなので無視
    except FileNotFoundError as e:
        print(e)
        print("必要な音声ファイルが見つかりません。プログラムを終了します。")
        return
    except ValueError as e:
        print(e)
        print("音声ファイルのサンプリングレートが一致しません。プログラムを終了します。")
        return

    print("音声データ読み込み完了")
    print(f"サンプリング周波数: {fs} Hz")
    print(f"マイクデータ形状: {MicData_combined.shape}")
    print(f"参照データ形状: {RefData_combined.shape}")

    # STFT変換を行う (プロット目的のため、ここでは最初のチャンネルのみ)
    # MicData_combinedは(サンプル数, チャンネル数)なので、[:, 0] で最初のチャンネルを選択
    MicData_combined_stft = ComputeSfft(MicData_combined)
    RefData_combined_stft = ComputeSfft(RefData_combined)

    # フィルタタイプの順列を生成
    # 'lowpass', 'highpass', 'bandpass' の3種類のフィルタの順列
    filter_types_base = ['lowpass', 'highpass', 'bandpass']
    FilterList_permutations = [list(p) for p in permutations(filter_types_base)]

    # フィルタ適用前のFFTをプロット (最初のチャンネルのみ)
    PlotFFT(MicData_combined[:, 0], fs, f"Mic (Before Filter) - Channel 1 FFT", f"fft_before_mic_ch1.png")
    PlotFFT(RefData_combined[:, 0], fs, f"Reference (Ground Truth) - Channel 1 FFT", f"fft_reference_ch1.png")

    print("\nSDR最適化探索開始...")
    # SutdyFilter の呼び出し
    best_filtered_output, best_config, best_score_full_tuple = SutdyFilter(
        FilterList_permutations, MicData_combined, RefData_combined, fs
    )
    
    if best_config:
        print("\n--- 最適化結果 ---")
        print(f"最適なフィルタ構成のタイプ: {best_config[0]}")
        print(f"最適なフィルタ構成の次数: {best_config[1]}")
        print(f"最適なフィルタ構成のカットオフ: {best_config[2]}")
        
        if best_score_full_tuple:
            # best_score_full_tuple は (obsSDR, estSDR, impSDR)
            print(f"観測SDR（フィルタ前）：{best_score_full_tuple[0]} [dB]")
            print(f"推定SDR（フィルタ後）：{best_score_full_tuple[1]} [dB]")
            print(f"SDR改善量（前後の変化）：{best_score_full_tuple[2]} [dB]")
            print(f"合計SDR改善量 (ImpSDR合計): {np.sum(best_score_full_tuple[2]):.4f} [dB]")

        # 最適なフィルタ適用後の音声を保存
        if best_filtered_output is not None:
            output_filename = "best_filtered_output.wav"
            sf.write(output_filename, best_filtered_output, fs)
            print(f"最適なフィルタ適用後の音声を {output_filename} に保存しました。")

            # 最適なフィルタ適用後のFFTをプロット (最初のチャンネルのみ)
            PlotFFT(best_filtered_output[:, 0], fs, f"Optimized Filtered Output - Channel 1 FFT", f"fft_after_optimized_ch1.png")

            # 最適なフィルタ構成の周波数応答をプロット
            print("\n最適なフィルタ構成の周波数応答をプロット中...")
            # 3つのフィルタを順に適用する構成なので、個々のフィルタ特性をプロット
            for i in range(len(best_config[0])):
                ftype = best_config[0][i]
                order = best_config[1][i]
                cutoff = best_config[2][i]
                b, a = CreateFilter(ftype, order, fs, cutoff)
                PlotFilterResponse(b, a, fs, f"最適フィルタ_{i+1}_{ftype}_O{order}_C{cutoff}")

    else:
        print("最適化処理が完了しませんでした。")

    print("\n--- フィルタ最適化処理終了 ---")


if __name__ == '__main__':
    main()