# 192GB DDR5 on AM5 — Stability Report

## Hardware Configuration

| Component | Model |
|-----------|-------|
| **CPU** | AMD Ryzen 9 9950X |
| **Motherboard** | Gigabyte B650 Eagle |
| **RAM** | 4× Corsair Vengeance RGB CMH96GX5M2B6400C32 (4×48GB = 192GB) |
| **Cooler** | Noctua NH-D15 (single fan config) |
| **Thermal Paste** | Thermal Grizzly Kryonaut |

## Cooler Configuration Note

The Corsair Vengeance RGB modules are tall (with RGB heatspreaders), which prevents mounting the second fan on the Noctua NH-D15 in standard dual-fan configuration.

**Workaround:**
- NH-D15 running in **single fan mode** (center-mounted)
- Second fan mounted as **case exhaust directly above the heatsink**
- Compensated with **Thermal Grizzly Kryonaut** thermal paste (high performance)

**Result:** Max CPU temperature of 58°C during overnight stress testing — the workaround works effectively.

## Noctua Mounting Tips for AM5

### Required Accessories

| Accessory | Model | Purpose |
|-----------|-------|---------|
| AM5 Mounting Kit | **Noctua NM-AM5/4-MP83** | Required for NH-D15 on AM5 socket |
| Anti-Vibration Mounts | **Noctua NA-SAV2** | For mounting displaced fan as case exhaust |

### Clearance Warning

If you have a case with limited CPU cooler clearance (like **Genesis IRID 505F** or similar ~165mm limit cases), get the **Noctua NA-SAV2** rubber anti-vibration mounts. They allow you to mount the second fan as a case exhaust fan above the heatsink instead of directly on the cooler.

### NH-D15 Mounting Order (Read the Manual!)

When installing Noctua NH-D15 with SecuFirm2 system on AM5, follow this order:

1. **Silicone spacers** go on the screw posts first (around the screw holes)
2. **Mounting bars** placed with the **arch/curve facing inward** (toward CPU)
3. **Thumbscrews** thread through mounting bars into the AMD backplate

**Important:** The stock AMD backplate stays on the motherboard — do not remove it. Noctua's SecuFirm2 system mounts directly to it.

## Achieved Settings

| Parameter | Value |
|-----------|-------|
| Memory Speed | **DDR5-5200 MHz** |
| Multiplier | 52.00 |
| Timings | CL32-40-40-84 |
| Command Rate | 1T |
| tRFC | 1311T |

## Voltage Configuration

| Voltage | Value |
|---------|-------|
| VDD (DRAM) | 1.35V |
| VDDQ | 1.35V |
| VDDIO | 1.35V |
| VDDP | 0.940V |
| VPP | 1.80V |
| VSOC | Auto |

## BIOS Settings

| Setting | Value |
|---------|-------|
| XMP Profile | **Enabled** |
| Memory Multiplier | 52.00 |
| UCLK DIV1 Mode | Auto |
| FCLK | Auto |
| Power Down Enable | Disabled |
| Memory Context Restore | Disabled |

## Key Findings

### XMP Works on AMD
The Corsair CMH96GX5M2B6400C32 is **Intel XMP only** (no AMD EXPO profile), but the XMP profile loaded successfully on the AMD B650 chipset. Manual frequency adjustment to 5200 MHz was required for stability with 4 DIMMs.

### Realistic Speed Expectations for 4×48GB
- **6400 MHz** — Not achievable with 4 DIMMs
- **5600 MHz** — Possible with luck, risky
- **5200 MHz** — Stable, recommended ✅
- **4800 MHz** — Guaranteed fallback

### Test Duration Warning
**TestMem5 with anta777 extreme config takes a long time with 192GB RAM.**

| RAM Capacity | Approximate Time per Cycle |
|--------------|---------------------------|
| 32 GB | 30-45 min |
| 64 GB | 1-1.5 hours |
| 128 GB | 2-3 hours |
| **192 GB** | **4-6 hours** |

Running 2 cycles overnight is normal and expected.

## Stability Verification

- ✅ TestMem5 (anta777 extreme): **2 cycles, 0 errors**
- ✅ CPU Max Temperature: **58°C**
- ✅ Windows detecting full 192GB

## Performance Notes

Difference between 5200 MHz and 5600 MHz is minimal (1-3%). For 192GB configurations, stability is more valuable than chasing extra speed.

---

**Author:** Norbert Marchewka  
**Date:** November 2025  
**Platform:** AMD AM5 / Zen 5
