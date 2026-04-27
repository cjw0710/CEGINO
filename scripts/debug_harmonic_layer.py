import torch
from neuralop.layers.harmonic_projector import HarmonicProjector
from neuralop.layers.harmonic_spectral_convolution import HarmonicSpectralConv


def main():
    B, C, K = 2, 8, 24
    x = torch.randn(B, C, K, K, K)
    x_fft = torch.fft.fftn(x.to(torch.cfloat), dim=(-3, -2, -1))

    projector = HarmonicProjector((K, K, K), lmax=2, radial_bins=8)
    coeffs = projector.project(projector.slice_fft(x_fft))

    conv = HarmonicSpectralConv(C, C, shell_count=projector.shell_count, lmax=2)
    coeffs_out = conv(coeffs)
    x_fft_out = projector.inverse(coeffs_out)

    print('coeff shapes:', [tuple(c.shape) for c in coeffs])
    print('reconstructed sliced fft:', tuple(x_fft_out.shape))


if __name__ == '__main__':
    main()
