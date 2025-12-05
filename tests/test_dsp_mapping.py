import numpy as np
import pytest
from commstools.dsp import mapping


def test_gray_code():
    n = 3
    codes = mapping.gray_code(n)
    assert len(codes) == 2**n
    assert len(set(codes)) == 2**n  # Unique

    # Check Gray property: adjacent elements differ by exactly 1 bit
    for i in range(len(codes)):
        c1 = codes[i]
        c2 = codes[(i + 1) % len(codes)]
        # xor gives 1s at differing positions
        xor = c1 ^ c2
        # Counting set bits
        diff_bits = bin(xor).count("1")
        assert diff_bits == 1, (
            f"Adjacent codes {bin(c1)} and {bin(c2)} differ by {diff_bits} bits"
        )


def test_qpsk_gray_mapping():
    qpsk = mapping.gray_constellation("psk", 4)
    assert len(qpsk) == 4

    # Expected QPSK with Gray mapping:
    # 00 (0) -> 1+0j (0 deg)
    # 01 (1) -> 0+1j (90 deg)
    # 11 (3) -> -1+0j (180 deg)
    # 10 (2) -> 0-1j (270 deg)
    # So the array at indices [0, 1, 2, 3] should be:
    # [1+0j, 0+1j, 0-1j, -1+0j]

    expected = np.array([1 + 0j, 0 + 1j, 0 - 1j, -1 + 0j])
    assert np.allclose(qpsk, expected)


def test_bpsk_real():
    bpsk = mapping.gray_constellation("psk", 2)
    assert len(bpsk) == 2
    # Should be strictly real float array, not complex
    assert bpsk.dtype == float or np.issubdtype(bpsk.dtype, np.floating)
    assert not np.iscomplexobj(bpsk)
    expected = np.array([-1.0, 1.0])
    assert np.allclose(bpsk, expected)


def test_ask_gray_mapping():
    # Formerly pam
    ask4 = mapping.gray_constellation("ask", 4)
    assert len(ask4) == 4
    # Gray: 0, 1, 3, 2
    # Points: -3, -1, 1, 3 (standard M-PAM usually centered)
    # M=4: -(4-1), -(4-3)... -> -3, -1, 1, 3

    # mapping[0] = -3
    # mapping[1] = -1
    # mapping[3] = 1 -> mapping[3] is the 3rd element in array ?? NO.

    # My implementation:
    # constellation[gray[i]] = points[i]
    # points = [-3, -1, 1, 3]
    # gray = [0, 1, 3, 2]
    # const[0] = -3
    # const[1] = -1
    # const[3] = 1
    # const[2] = 3

    expected = np.array([-3, -1, 3, 1])
    assert np.allclose(ask4, expected)


def test_qam16_shape():
    qam16 = mapping.gray_constellation("qam", 16)
    assert len(qam16) == 16
    assert not np.isrealobj(qam16)  # Should be complex


def test_cross_qam_32():
    qam32 = mapping.gray_constellation("qam", 32)
    assert len(qam32) == 32

    # Check bounds (Cross shape range -5..5 for 32-QAM)
    assert np.all(np.abs(qam32.real) <= 5.0001)
    assert np.all(np.abs(qam32.imag) <= 5.0001)

    # Check that corners are empty
    corners = [5 + 5j, 5 - 5j, -5 + 5j, -5 - 5j]
    for c in corners:
        dists = np.abs(qam32 - c)
        assert np.min(dists) > 0.1

    assert len(np.unique(qam32)) == 32


def test_cross_qam_128():
    qam128 = mapping.gray_constellation("qam", 128)
    assert len(qam128) == 128

    # Check standard cross bounds for 128-QAM (12x12 minus corners)
    # Range -11..11
    assert np.all(np.abs(qam128.real) <= 11.0001)
    assert np.all(np.abs(qam128.imag) <= 11.0001)

    assert len(np.unique(qam128)) == 128


def test_cross_qam_8_error():
    # 8-QAM is no longer supported
    with pytest.raises(ValueError, match="8-QAM is not supported"):
        mapping.gray_constellation("qam", 8)
