import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import fftconvolve
from itertools import permutations # 忘れないようにインポート


def alignment(x, y):
    """
    自己相関関数を用いた信号のアラインメント（時間ずれ補正）
    xとyが入力だがyの時間をずらして合わせる（のでxは何もせず出力する）
    """
    sigLen, nCh = x.shape
    y_out = np.zeros_like(y)
    delay = np.zeros(nCh, dtype=int)

    for n in range(nCh):
        # 相互相関関数の計算
        c = np.correlate(y[:, n], x[:, n], mode='full')
        lags = np.arange(-len(y[:, n]) + 1, len(x[:, n]))

        # 相互相関関数の最大値となるインデックスを取得
        max_idx = np.argmax(c)
        delay[n] = lags[max_idx]  # 波形の時間ずれ量を計算

        if delay[n] > 0:  # 正の時間ずれを補正
            y_out[:-delay[n], n] = y[delay[n]:, n]
            y_out[-delay[n]:, n] = 0
        elif delay[n] < 0:  # 負の時間ずれを補正
            y_out[-delay[n]:, n] = y[:y.shape[0] + delay[n], n]
            y_out[:-delay[n], n] = 0
        else:  # 時間ずれ無し
            y_out[:, n] = y[:, n]
    
    x_out = x  # xは何もせず出力
    return x_out, y_out, delay

def _bss_decomp_mtifilt(se, s, j, flen):
    """
    BSS_DECOMP_MTIFILT Decomposition of an estimated source image into four
    components representing respectively the true source image, spatial (or
    filtering) distortion, interference and artifacts, derived from the true
    source images using multichannel time-invariant filters.
    """
    nchan2, nsampl2 = se.shape
    nsrc, nsampl, nchan = s.shape

    if not (nchan2 == nchan and nsampl2 == nsampl):
        raise ValueError("Dimensions of se and s are not compatible.")

    # True source image
    s_true = np.concatenate((s[j, :, :].T, np.zeros((nchan, flen - 1))), axis=1)

    # Spatial (or filtering) distortion
    e_spat = _project(se, s[j:j+1, :, :], flen) - s_true

    # Interference
    e_interf = _project(se, s, flen) - s_true - e_spat

    # Artifacts
    e_artif = np.concatenate((se, np.zeros((nchan, flen - 1))), axis=1) - s_true - e_spat - e_interf

    return s_true, e_spat, e_interf, e_artif

def _project(se, s, flen):
    """
    SPROJ Least-squares projection of each channel of se on the subspace
    spanned by delayed versions of the channels of s, with delays between 0
    and flen-1
    """
    nsrc, nsampl, nchan = s.shape
    s_reshaped = s.transpose(2, 0, 1).reshape(nchan * nsrc, nsampl)

    # Zero padding and FFT of input data
    s_padded = np.concatenate((s_reshaped, np.zeros((nchan * nsrc, flen - 1))), axis=1)
    se_padded = np.concatenate((se, np.zeros((nchan, flen - 1))), axis=1)
    
    fftlen = 2**np.ceil(np.log2(nsampl + flen - 1)).astype(int)
    sf = np.fft.fft(s_padded, n=fftlen, axis=1)
    sef = np.fft.fft(se_padded, n=fftlen, axis=1)

    # Inner products between delayed versions of s
    G = np.zeros((nchan * nsrc * flen, nchan * nsrc * flen))
    for k1 in range(nchan * nsrc):
        for k2 in range(k1 + 1): # MATLAB's toeplitz behavior
            ssf = sf[k1, :] * np.conj(sf[k2, :])
            ssf = np.real(np.fft.ifft(ssf))
            
            # Constructing Toeplitz matrix based on MATLAB's toeplitz(c,r)
            # c = ssf[0:flen]
            # r = np.concatenate(([ssf[0]], ssf[fftlen - flen + 2 -1 : fftlen][::-1])) # Correct for 0-indexing
            r_matlab = np.concatenate((ssf[0:1], ssf[fftlen-flen+1:fftlen][::-1]))
            c_matlab = ssf[0:flen]
            
            # Manual Toeplitz construction to match MATLAB's behavior with specific indices
            ss = np.zeros((flen, flen))
            for row in range(flen):
                for col in range(flen):
                    idx = col - row
                    if 0 <= idx < flen:
                        ss[row, col] = c_matlab[idx]
                    elif -flen < idx < 0:
                        ss[row, col] = r_matlab[-idx] # Corresponds to MATLAB's ssf([1 fftlen:-1:fftlen-flen+2]) reversed
            
            G[k1 * flen : (k1 + 1) * flen, k2 * flen : (k2 + 1) * flen] = ss
            if k1 != k2:
                G[k2 * flen : (k2 + 1) * flen, k1 * flen : (k1 + 1) * flen] = ss.T

    # Inner products between se and delayed versions of s
    D = np.zeros((nchan * nsrc * flen, nchan))
    for k in range(nchan * nsrc):
        for i in range(nchan):
            ssef = sf[k, :] * np.conj(sef[i, :])
            ssef = np.real(np.fft.ifft(ssef, axis=0))
            D[k * flen : (k + 1) * flen, i] = ssef[[0] + list(range(fftlen - 1, fftlen - flen - 1, -1))].T # Equivalent to MATLAB's ssef(:,[1 fftlen:-1:fftlen-flen+2]).'


    # Distortion filters
    C = np.linalg.solve(G, D)
    C = C.reshape(flen, nchan * nsrc, nchan)

    # Filtering
    sproj = np.zeros((nchan, nsampl + flen - 1))
    for k in range(nchan * nsrc):
        for i in range(nchan):
            # Equivalent to MATLAB's fftfilt
            # Here we can use scipy.signal.convolve or manually do FFT-based convolution
            sproj[i, :] += fftconvolve(s_reshaped[k, :], C[:, k, i].flatten(), mode='full')[:nsampl + flen -1] # Ensure output length matches MATLAB

    return sproj

def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    """
    BSS_SOURCE_CRIT Measurement of the separation quality for a given source
    in terms of filtered true source, interference and artifacts.
    """
    if not (s_true.shape == e_spat.shape == e_interf.shape == e_artif.shape):
        raise ValueError("All components must have the same dimensions.")

    s_filt = s_true + e_spat
    
    # SDR
    sdr = 10 * np.log10(np.sum(s_filt**2) / np.sum((e_interf + e_artif)**2))
    # SIR
    sir = 10 * np.log10(np.sum(s_filt**2) / np.sum(e_interf**2))
    # SAR
    sar = 10 * np.log10(np.sum((s_filt + e_interf)**2) / np.sum(e_artif**2))

    return sdr, sir, sar

def bss_eval_sources(se, s):
    """
    BSS_EVAL_SOURCES Ordering and measurement of the separation quality for
    estimated source signals in terms of filtered true source, interference
    and artifacts.
    """
    nsrc, nsampl = se.shape
    nsrc2, nsampl2 = s.shape

    if not (nsrc2 == nsrc and nsampl2 == nsampl):
        raise ValueError("The number of estimated sources and reference sources must be equal and have the same duration.")

    # Ensure s has nsrc x nsampl x nchan structure for _bss_decomp_mtifilt
    # Assuming s is nsrc x nsampl and se is nsrc x nsampl in the input,
    # but the internal functions expect nchan. Let's assume nchan = 1 for simplicity if not provided.
    # From MATLAB code: s: nsrc x nsampl matrix containing true sources
    # _bss_decomp_mtifilt expects s: nsrc x nsampl x nchan
    # This implies s should be reshaped or assumed to have a single channel.
    # Let's assume s and se are already correctly formatted or will be handled externally
    # Based on main_eval.m, ref and est are (samples x sources).
    # bss_eval_sources(obs.', ref.') means (sources x samples).
    # In bss_decomp_mtifilt: se: nchan x nsampl, s: nsrc x nsampl x nchan
    # This means se is (1 x nsampl) and s is (nsrc x nsampl x 1)
    
    # Let's reshape s to have a channel dimension of 1 if it's 2D
    if s.ndim == 2:
        s = s[:, :, np.newaxis]
    if se.ndim == 1:
        se = se[np.newaxis, :] # Add channel dim if mono and passed as 1D

    # Performance criteria
    SDR = np.zeros((nsrc, nsrc))
    SIR = np.zeros((nsrc, nsrc))
    SAR = np.zeros((nsrc, nsrc))
    
    # The flen (filter length) in bss_decomp_mtifilt is hardcoded to 512 in MATLAB's bss_eval_sources.m
    flen = 512 

    for jest in range(nsrc):
        for jtrue in range(nsrc):
            # Ensure se for _bss_decomp_mtifilt is (nchan x nsampl). Here nchan=1
            s_true, e_spat, e_interf, e_artif = _bss_decomp_mtifilt(se[jest:jest+1, :], s, jtrue, flen)
            SDR[jest, jtrue], SIR[jest, jtrue], SAR[jest, jtrue] = _bss_source_crit(s_true, e_spat, e_interf, e_artif)

    # Selection of the best ordering
    perms = np.array(list(permutations(range(nsrc))))
    meanSIR = np.zeros(len(perms))
    for p_idx, p in enumerate(perms):
        # Flattening to match MATLAB's linear indexing: (0:nsrc-1)*nsrc+perm(p,:)
        linear_indices = np.arange(nsrc) * nsrc + p
        meanSIR[p_idx] = np.mean(SIR.flatten()[linear_indices])

    popt = np.argmax(meanSIR)
    perm = perms[popt]

    # Reorder results based on best permutation
    SDR_out = np.array([SDR[i, perm[i]] for i in range(nsrc)])
    SIR_out = np.array([SIR[i, perm[i]] for i in range(nsrc)])
    SAR_out = np.array([SAR[i, perm[i]] for i in range(nsrc)])

    return SDR_out, SIR_out, SAR_out, perm


def GetScorePoint(ref_data_np, mic_data_np, est_data_np, fs=None): # fsを追加して、内部で必要なら使う
    """
    Calculates the Signal to Distortion Ratio (SDR) for digital filter
    performance in suppressing overlapping sounds, accepting NumPy arrays directly.

    Args:
        ref_data_np (np.ndarray): True source signals (samples x channels).
        mic_data_np (np.ndarray): Observed signals before filtering (samples x channels).
        est_data_np (np.ndarray): Signals after filter processing (samples x channels).
        fs (int, optional): Sampling rate. Not strictly used here if bss_eval_sources
                            doesn't need it, but good practice for audio data.

    Returns:
        tuple: A tuple containing:
            - obsSDR (np.array): SDR values for observed signals (before filtering).
            - estSDR (np.array): SDR values for estimated signals (after filtering).
            - impSDR (np.array): SDR improvement (performance of overlap suppression).
    """
    # ref_data_np, mic_data_np, est_data_np は既に (samples x channels) 形式であることを想定

    # refとestの間で波形の時間ずれがある場合に補正するアラインメント処理
    # alignment は (samples, channels) を受け取り (samples, channels) を返す
    ref_aligned, est_aligned, delay = alignment(ref_data_np, est_data_np)
    
    # alignment は mic_data_np と ref_data_np の間でも行うべきか？
    # MATLABの `main_eval.m` では `[ref, est, delay] = alignment(ref, est);`
    # そして `obsSDR = bss_eval_sources(obs.', ref.');`
    # `estSDR = bss_eval_sources(est.', ref.');` となっているため、
    # `obs` と `ref` のアラインメントは行われていません。
    # したがって、`mic_data_np` はアラインメントしないまま `bss_eval_sources` に渡します。

    print("SDR計算中...")
    # bss_eval_sources expects (nsrc x nsampl) for se and s
    # NumPy配列は (samples x channels) なので、transpose (.T) して渡す
    obsSDR, _, _, _ = bss_eval_sources(mic_data_np.T, ref_data_np.T) # 観測信号のSDR
    estSDR, _, _, _ = bss_eval_sources(est_aligned.T, ref_aligned.T) # 推定信号のSDR
    impSDR = estSDR - obsSDR # SDR改善量（フィルタ処理で得られた被り音抑圧の性能）

    return obsSDR, estSDR, impSDR