import numpy as np
import torch

# --- Các hằng số cho pipeline kỹ thuật số ---
CODE_RATE = 0.5  # Ví dụ: LDPC rate 1/2
BITS_PER_SYMBOL = 4 # Ví dụ: 16-QAM
MODULATION_ORDER = 16

# ... (Sao chép các hàm ldpc_encode, ldpc_decode, modulate_16qam, 
#       demodulate_16qam, simulate_rayleigh_channel từ câu trả lời trước) ...

def indices_to_bitstream(indices: np.ndarray, num_bits: int) -> np.ndarray:
    """Chuyển đổi một mảng chỉ số thành một chuỗi bit."""
    bitstream_list = [format(idx, f'0{num_bits}b') for idx in indices]
    bitstream_str = "".join(bitstream_list)
    return np.array([int(b) for b in bitstream_str])

def bitstream_to_indices(bitstream: np.ndarray, num_bits: int, num_indices: int) -> np.ndarray:
    """Chuyển đổi chuỗi bit về lại mảng chỉ số."""
    indices = []
    num_bits_total = len(bitstream)
    for i in range(0, num_bits_total, num_bits):
        if i + num_bits <= num_bits_total:
            bit_chunk_str = "".join(map(str, bitstream[i:i+num_bits]))
            try:
                indices.append(int(bit_chunk_str, 2))
            except ValueError:
                indices.append(0) # Nếu có lỗi, trả về chỉ số 0
    
    # Cắt hoặc đệm để đảm bảo đúng số lượng chỉ số
    if len(indices) > num_indices:
        indices = indices[:num_indices]
    elif len(indices) < num_indices:
        indices.extend([0] * (num_indices - len(indices)))
        
    return np.array(indices)

# --- Các hàm chính để gọi ---

def transmit_indices(indices_tensor: torch.Tensor, num_bits: int):
    """Thực hiện các bước phía transmitter."""
    indices_np = indices_tensor.cpu().numpy().flatten()
    
    # 1. Pack bits
    source_bits = indices_to_bitstream(indices_np, num_bits)
    
    # 2. Channel Coding
    coded_bits = ldpc_encode(source_bits, CODE_RATE)
    
    # 3. Modulation
    symbols = modulate_16qam(coded_bits)
    
    # Lưu lại độ dài gốc để giải mã
    return symbols, len(source_bits)

def receive_indices(tx_output: tuple, snr_db: float, num_original_indices: int):
    """Thực hiện các bước phía receiver."""
    symbols, original_bit_len = tx_output
    num_bits = original_bit_len // num_original_indices

    # 4. Pass through channel
    noisy_symbols = simulate_rayleigh_channel(symbols, snr_db)
    
    # 5. Demodulation
    demodulated_bits = demodulate_16qam(noisy_symbols)
    
    # 6. Channel Decoding
    recovered_source_bits = ldpc_decode(demodulated_bits, CODE_RATE)
    
    # Cắt bớt phần padding có thể có từ LDPC encode
    if len(recovered_source_bits) > original_bit_len:
        recovered_source_bits = recovered_source_bits[:original_bit_len]

    # 7. Unpack bits to indices
    recovered_indices = bitstream_to_indices(recovered_source_bits, num_bits, num_original_indices)
    
    return torch.from_numpy(recovered_indices).long()

def transmit_and_receive_indices_batch(indices_tensor: torch.Tensor, vq_bits_map: torch.Tensor, snr_db: float, code_rate=0.5):
    """
    Mô phỏng pipeline truyền kỹ thuật số cho một batch.
    """
    device = indices_tensor.device
    batch_size, num_patches = indices_tensor.shape
    
    indices_np = indices_tensor.cpu().numpy()
    vq_bits_np = vq_bits_map.cpu().numpy()
    
    recovered_indices_batch_list = []
    
    for i in range(batch_size): # Lặp qua từng ảnh
        # 1. Bit Packing
        bitstream_list = [format(idx, f'0{bits}b') for idx, bits in zip(indices_np[i], vq_bits_np[i])]
        source_bitstream_str = "".join(bitstream_list)
        source_bitstream = np.array([int(b) for b in source_bitstream_str])
        
        # 2. Channel Coding
        coded_bitstream = ldpc_encode(source_bitstream, code_rate)
        
        # 3. Modulation
        modulated_symbols = modulate_16qam(coded_bitstream)

        # 4. Channel
        received_symbols = simulate_rayleigh_channel(modulated_symbols, snr_db)

        # 5. Demodulation
        demod_bits = demodulate_16qam(received_symbols)
        demod_bits = demod_bits[:len(coded_bitstream)] # Cắt padding

        # 6. Channel Decoding
        recovered_bits = ldpc_decode(demod_bits, code_rate)
        recovered_bits = recovered_bits[:len(source_bitstream)] # Cắt padding

        # 7. Bit Unpacking
        recovered_indices_np = bitstream_to_indices(recovered_bits, vq_bits_np[i], num_patches)
        recovered_indices_batch_list.append(torch.from_numpy(recovered_indices_np).long())

    return torch.stack(recovered_indices_batch_list).to(device)

def bitstream_to_indices(bitstream: np.ndarray, bits_map: np.ndarray, num_indices: int) -> np.ndarray:
    """Chuyển đổi chuỗi bit về lại mảng chỉ số, với số bit thay đổi cho mỗi chỉ số."""
    indices = []
    current_pos = 0
    for i in range(num_indices):
        num_bits = bits_map[i]
        if current_pos + num_bits <= len(bitstream):
            bit_chunk_str = "".join(map(str, bitstream[current_pos : current_pos + num_bits]))
            indices.append(int(bit_chunk_str, 2))
            current_pos += num_bits
        else:
            indices.append(0) # Nếu hết bit, pad bằng index 0
    return np.array(indices)
