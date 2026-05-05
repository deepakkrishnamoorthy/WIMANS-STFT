from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("stft_top5_multihead_architecture.png")
W, H = 2400, 1380
S = W / 1600


def font(size, bold=False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, int(size * S))
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_SUB = font(18)
F_SECTION = font(18, True)
F_BOX_TITLE = font(21, True)
F_BOX = font(16)
F_TINY = font(14)


def xy(v):
    return tuple(int(round(x * S)) for x in v)


def rounded(draw, box, fill, outline="#405169", width=2, radius=12):
    draw.rounded_rectangle(xy(box), radius=int(radius * S), fill=fill, outline=outline, width=int(width * S))


def text(draw, pos, content, fill="#172033", fnt=F_BOX):
    draw.text(xy(pos), content, fill=fill, font=fnt)


def arrow(draw, start, end, fill="#2f3a4a", width=3):
    draw.line([xy(start), xy(end)], fill=fill, width=int(width * S))
    x1, y1 = xy(start)
    x2, y2 = xy(end)
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    size = int(11 * S)
    pts = [
        (x2, y2),
        (int(x2 - size * math.cos(angle - 0.45)), int(y2 - size * math.sin(angle - 0.45))),
        (int(x2 - size * math.cos(angle + 0.45)), int(y2 - size * math.sin(angle + 0.45))),
    ]
    draw.polygon(pts, fill=fill)


def curved_arrow(draw, points, fill="#2f3a4a", width=3):
    scaled = [xy(p) for p in points]
    draw.line(scaled, fill=fill, width=int(width * S), joint="curve")
    arrow(draw, points[-2], points[-1], fill=fill, width=0.1)


def box(draw, rect, title, lines, fill):
    rounded(draw, rect, fill)
    x, y, _, _ = rect
    text(draw, (x + 25, y + 27), title, fnt=F_BOX_TITLE)
    for i, line in enumerate(lines):
        text(draw, (x + 25, y + 58 + i * 23), line, fill="#435267" if i == 0 else "#5b677a", fnt=F_BOX if i == 0 else F_TINY)


def main():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    text(d, (70, 66), "STFT Top-5 Multi-Head CLSTM Architecture", fnt=F_TITLE)
    text(d, (72, 98), "Compact WiFi CSI representation with shared temporal feature learning and task-specific supervision", fill="#4e5b6e", fnt=F_SUB)

    # Section 1
    rounded(d, (55, 130, 1545, 315), "#f8fafc", "#d8dee8", 1.5, 14)
    text(d, (75, 160), "Input And STFT Representation", fill="#334155", fnt=F_SECTION)
    box(d, (85, 185, 340, 275), "Raw CSI Amplitude", ["WiMANS sample", "3 Tx x 3 Rx x 30 subcarriers"], "#e9f5ff")
    box(d, (405, 185, 665, 275), "Top-5 Subcarriers", ["Keep informative carriers", "3 x 3 x 5 = 45 CSI streams"], "#e9f5ff")
    box(d, (730, 185, 1005, 275), "STFT Extraction", ["Time-frequency transform", "Per selected CSI stream"], "#eefaf3")
    box(d, (1070, 185, 1375, 275), "45-Channel STFT Tensor", ["Input to neural model", "C x F x T = 45 x 129 x 200"], "#eefaf3")
    arrow(d, (340, 230), (405, 230))
    arrow(d, (665, 230), (730, 230))
    arrow(d, (1005, 230), (1070, 230))

    # Section 2
    rounded(d, (55, 350, 1545, 580), "#f8fafc", "#d8dee8", 1.5, 14)
    text(d, (75, 380), "Shared Feature Extractor", fill="#334155", fnt=F_SECTION)
    box(d, (115, 420, 375, 520), "Normalization", ["log1p + per-channel standardization", "Training augmentation optional"], "#eefaf3")
    box(d, (455, 420, 730, 520), "ResNet18 Backbone", ["2D convolutional feature extraction", "First conv adapted to 45 channels"], "#fff5df")
    box(d, (815, 420, 1090, 520), "Temporal Sequence", ["Mean over frequency axis", "Feature map to time sequence"], "#fff5df")
    box(d, (1175, 420, 1450, 520), "BiLSTM Temporal Model", ["Contextual temporal aggregation", "Pooled shared representation z"], "#fff5df")
    arrow(d, (1375, 230), (1450, 470))
    arrow(d, (375, 470), (455, 470))
    arrow(d, (730, 470), (815, 470))
    arrow(d, (1090, 470), (1175, 470))

    # Section 3
    rounded(d, (55, 615, 1545, 825), "#f8fafc", "#d8dee8", 1.5, 14)
    text(d, (75, 645), "Multi-Head Supervision", fill="#334155", fnt=F_SECTION)
    box(d, (115, 685, 420, 780), "Activity-Set Head", ["9 sigmoid outputs", "Which activities appear in the scene?"], "#f0ecff")
    box(d, (490, 685, 795, 780), "Occupancy Head", ["6 sigmoid outputs", "Which user slots are active?"], "#ffeef2")
    box(d, (865, 685, 1170, 780), "User-Slot Activity Head", ["54 sigmoid outputs", "6 user slots x 9 activities"], "#eef2ff")
    box(d, (1240, 675, 1495, 790), "Training Objective", ["Weighted BCE losses", "+ occupancy consistency", "+ active-count regularizer"], "#f7f8fb")
    arrow(d, (1312, 520), (267, 685))
    arrow(d, (1312, 520), (642, 685))
    arrow(d, (1312, 520), (1017, 685))
    arrow(d, (1170, 733), (1240, 733))

    rounded(d, (90, 850, 1510, 895), "#f8fafc", "#d8dee8", 1.2, 10)
    text(
        d,
        (115, 878),
        "Evaluation reports activity-set F1, occupancy F1, active-slot F1, 54-label F1, and class-wise confusion diagnostics. Validation selects the best checkpoint; test is evaluated once.",
        fill="#5b677a",
        fnt=F_TINY,
    )

    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
