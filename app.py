import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
from solver import solve

# Optional (Windows users):
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Sudoku Solver", layout="centered")
st.title("🧠 AI Sudoku Solver (Upload + Webcam + Editable)")

# ---------- Image Preprocessing ----------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

# ---------- Digit Extraction ----------
def extract_digits(img):
    board = np.zeros((9, 9), dtype=int)
    mask = np.zeros((9, 9), dtype=bool)
    h, w = img.shape
    ch, cw = h // 9, w // 9
    config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'

    for i in range(9):
        for j in range(9):
            y1, y2 = i * ch, (i + 1) * ch
            x1, x2 = j * cw, (j + 1) * cw
            cell = img[y1:y2, x1:x2]
            cell = cv2.resize(cell, (64, 64))
            cell = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

            contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digit_img = np.zeros_like(cell)
            for cnt in contours:
                x, y, w_, h_ = cv2.boundingRect(cnt)
                if w_ * h_ > 100:
                    digit_img = cell[y:y+h_, x:x+w_]
                    digit_img = cv2.resize(digit_img, (28, 28))
                    break
            else:
                digit_img = cell

            text = pytesseract.image_to_string(digit_img, config=config)
            text = ''.join(filter(str.isdigit, text))

            if text.isdigit():
                board[i][j] = int(text)
                mask[i][j] = True

    return board, mask

# ---------- Overlay Solution ----------
def overlay_solution(img, mask, solved):
    output = img.copy()
    h, w = output.shape[:2]
    ch, cw = h // 9, w // 9
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2

    for i in range(9):
        for j in range(9):
            if not mask[i][j]:
                val = str(solved[i][j])
                sz = cv2.getTextSize(val, font, scale, thickness)[0]
                x = j * cw + (cw - sz[0]) // 2
                y = i * ch + (ch + sz[1]) // 2
                cv2.putText(output, val, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return output

# ---------- Select Input Method ----------
method = st.radio("Choose input method:", ["📤 Upload Image", "📷 Use Webcam"])

# ---------- Upload Mode ----------
if method == "📤 Upload Image":
    uploaded = st.file_uploader("Upload Sudoku Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img = np.array(image)
        gray = preprocess(img)
        board, mask = extract_digits(gray)

        st.subheader("✏️ Edit Extracted Sudoku (if needed)")
        updated_board = np.zeros((9, 9), dtype=int)

        with st.form("edit_form"):
            for i in range(9):
                cols = st.columns(9)
                for j in range(9):
                    val = str(board[i][j]) if board[i][j] != 0 else ""
                    user_input = cols[j].text_input(f"{i}-{j}", val, max_chars=1, label_visibility="collapsed")
                    updated_board[i][j] = int(user_input) if user_input.isdigit() else 0
            submit = st.form_submit_button("🧠 Solve Sudoku")

        if submit:
            original_mask = (updated_board != 0)
            board_copy = updated_board.copy()
            if solve(board_copy):
                st.success("✅ Puzzle Solved!")
                result_img = overlay_solution(img.copy(), original_mask, board_copy)
                st.image(result_img, caption="Solved Sudoku", channels="BGR")
            else:
                st.error("❌ Could not solve the puzzle. Please check the entries.")

# ---------- Webcam Mode ----------
elif method == "📷 Use Webcam":
    st.warning("⚠️ Webcam works only in local environment.")
    go = st.checkbox("Start Camera")
    frame_holder = st.image([])

    cap = cv2.VideoCapture(0)
    while go:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to access camera.")
            break

        gray = preprocess(frame)
        board, mask = extract_digits(gray)
        board_copy = board.copy()

        if solve(board_copy):
            result = overlay_solution(frame, mask, board_copy)
            frame_holder.image(result, channels="BGR")
        else:
            frame_holder.image(frame, caption="❌ Sudoku not detected or unsolvable", channels="BGR")
    cap.release()
