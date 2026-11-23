import sys
import torch

print("=" * 40)
print("TEST DISPOSITIVO GRAFICO (CUDA)")
print("=" * 40)

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f" Scheda Video Rilevata: {device_name}")
    print(
        f" Memoria VRAM Totale: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    )
    print(" CUDA è attivo e pronto all'uso!")
else:
    print(" ATTENZIONE: CUDA non è attivo.")
    print("Whisper funzionerà sulla CPU (molto più lento).")
    print(
        "Rinstalla PyTorch con CUDA tramite: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    )

print("=" * 40)