import flet as ft
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def cv2_to_bytes(img: np.ndarray) -> bytes:
    """Chuyển ảnh OpenCV (BGR) sang bytes PNG để Flet hiển thị."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def main(page: ft.Page):
    page.title = "Ứng dụng Xử lý Ảnh với Flet + OpenCV"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window_width = 800
    page.window_height = 500

    # Biến toàn cục lưu ảnh gốc và ảnh đang xử lý
    state = {"orig": None, "proc": None}

    # Khung hiển thị ảnh
    img_control = ft.Image(src_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",width=500, height=400)

    # Hàm xử lý khi người dùng chọn file
    def pick_result(e: ft.FilePickerResultEvent):
        if e.files:
            f = e.files[0]
            # Đọc ảnh từ đường dẫn
            with open(f.path, "rb") as file:
                data = file.read()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Lưu state
            state["orig"] = img.copy()
            state["proc"] = img.copy()
            # Cập nhật lên UI
            img_bytes = cv2_to_bytes(img)
            img_control.src_base64 = base64.b64encode(img_bytes).decode()
            page.update()

    # Chọn File
    file_picker = ft.FilePicker(on_result=pick_result)
    page.overlay.append(file_picker)

    # Các chức năng xử lý ảnh
    def mean_filter(img):
        m, n = img.shape
        img_new = np.zeros([m, n])
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                img_new[i, j] = np.mean(img[i-1:i+2, j-1:j+2])
        return img_new.astype(np.uint8)

    def Mean_Filter(e):
        if state["proc"] is None:
            return
        gray = cv2.cvtColor(state["proc"], cv2.COLOR_BGR2GRAY)
        mean_img = mean_filter(gray)
        mean_img_color = cv2.cvtColor(mean_img, cv2.COLOR_GRAY2BGR)
        state["proc"] = mean_img_color
        # Đẩy lên UI
        img_bytes = cv2_to_bytes(state["proc"])
        img_control.src_base64 = base64.b64encode(img_bytes).decode()
        page.update()
    def median_filter(img):
        '''Lọc trung vi'''
        m, n = img.shape
        img_new = np.zeros([m, n])
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = sorted(img[i-1:i+2, j-1:j+2].flatten())
                img_new[i, j] = temp[4]
        return img_new.astype(np.uint8)

    def Median_Filter(e):
        if state["proc"] is None:
            return
        gray = cv2.cvtColor(state["proc"], cv2.COLOR_BGR2GRAY)
        median_img = median_filter(gray)
        median_img_color = cv2.cvtColor(median_img, cv2.COLOR_GRAY2BGR)
        state["proc"] = median_img_color
        #Đẩy lên UI
        img_bytes = cv2_to_bytes(state["proc"])
        img_control.src_base64 = base64.b64encode(img_bytes).decode()
        page.update()

    def Loc_Midpoint(img,ksize):
        m, n = img.shape
        img_new = np.zeros([m, n])
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                window = img[i-1:i+2, j-1:j+2].flatten()
                min_val = np.min(window)
                max_val = np.max(window)
                img_new[i, j] = (min_val + max_val) / 2
        return img_new.astype(np.uint8)
    
    def Loc_tktt_midpoint(e):
        if state["proc"] is None:
            return
        gray = cv2.cvtColor(state["proc"], cv2.COLOR_BGR2GRAY)
        Loc_Midpoint_img = Loc_Midpoint(gray,ksize=5)
        Loc_Midpoint_img_color = cv2.cvtColor(Loc_Midpoint_img, cv2.COLOR_GRAY2BGR)
        state["proc"] = Loc_Midpoint_img_color
        #Đẩy lên UI
        img_bytes = cv2_to_bytes(state["proc"])
        img_control.src_base64 = base64.b64encode(img_bytes).decode()
        page.update()

    def Loc_TK_Alpha_Trimmed(img, ksize, alpha):
        m,n = img.shape
        img_LQ_Trung_binh_cat_Alpha = np.zeros([m, n]).astype(float)
        h = (ksize - 1) // 2
        d = int(ksize*ksize*alpha)
        padded_img = np.pad(img,(h,h),mode='reflect')
        for i in range(m):
            for j in range(n):
                vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize].flatten()
                vung_anh_kich_thuoc_k.sort()
                vung_anh_kich_thuoc_con_lai = vung_anh_kich_thuoc_k[d//2:-d//2]
                img_LQ_Trung_binh_cat_Alpha[i,j] = np.sum(vung_anh_kich_thuoc_con_lai) / (ksize**2-d)
        return np.uint8(img_LQ_Trung_binh_cat_Alpha)
    
    def loc_tk_alpha_trimmed(e):
        if state["proc"] is None: return
        gray = cv2.cvtColor(state["proc"], cv2.COLOR_BGR2GRAY)
        alpha_img = Loc_TK_Alpha_Trimmed(gray,ksize=5,alpha=0.25)
        alpha_img_color = cv2.cvtColor(alpha_img, cv2.COLOR_GRAY2BGR)
        state["proc"] = alpha_img_color
        img_bytes = cv2_to_bytes(state["proc"])
        img_control.src_base64 = base64.b64encode(img_bytes).decode()
        page.update()

    def loc_trung_vi_thich_nghi(img, max_ksize=7):
        m, n = img.shape
        img_ket_qua = np.copy(img)
        padded = np.pad(img, (max_ksize//2, max_ksize//2), mode='reflect')
        for i in range(m):
            for j in range(n):
                for k in range(3, max_ksize+1, 2):
                    region = padded[i:i+k, j:j+k].flatten()
                    Zmed = np.median(region)
                    Zmin = np.min(region)
                    Zmax = np.max(region)
                    A1 = Zmed - Zmin
                    A2 = Zmed - Zmax
                    if A1 > 0 and A2 < 0:
                        B1 = img[i,j] - Zmin
                        B2 = img[i,j] - Zmax
                        if B1 > 0 and B2 < 0:
                            img_ket_qua[i,j] = img[i,j]
                        else:
                            img_ket_qua[i,j] = Zmed
                        break
                    elif k == max_ksize:
                        img_ket_qua[i,j] = Zmed
        return np.uint8(img_ket_qua)
    
    def Loc_Trung_Vi_Thich_Nghi(e):
        if state["proc"] is None:
            return
        gray = cv2.cvtColor(state["proc"], cv2.COLOR_BGR2GRAY)
        trungvithichnghi_img = loc_trung_vi_thich_nghi(gray)
        trungvithichnghi_img_color = cv2.cvtColor(trungvithichnghi_img, cv2.COLOR_GRAY2BGR)
        state["proc"] = trungvithichnghi_img_color
        #Đẩy lên UI
        img_bytes = cv2_to_bytes(state["proc"])
        img_control.src_base64 = base64.b64encode(img_bytes).decode()
        page.update()

    def loc_thich_nghi_cuc_bo(img, ksize, phuong_sai_nhieu):
        m, n = img.shape
        img_ket_qua_anh_loc = np.zeros([m, n]).astype(float)
        h=(ksize -1) // 2
        padded_img = np.pad(img, (h, h), mode='reflect')
        for i in range(m):
            for j in range(n):
                vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
                phuong_sai_cuc_bo = np.var(vung_anh_kich_thuoc_k)
                gia_tri_TB_cuc_bo = np.mean(vung_anh_kich_thuoc_k)
                if gia_tri_TB_cuc_bo > phuong_sai_nhieu :
                    img_ket_qua_anh_loc[i,j] = gia_tri_TB_cuc_bo
                else:
                    img_ket_qua_anh_loc[i,j] = padded_img[i,j] - int((phuong_sai_nhieu /
                        phuong_sai_cuc_bo) * (padded_img[i,j] - gia_tri_TB_cuc_bo))
        return np.uint8(img_ket_qua_anh_loc)

    def Loc_Thich_Nghi_Cuc_Bo(e):
        if state["proc"] is None:
            return
        gray = cv2.cvtColor(state["proc"], cv2.COLOR_BGR2GRAY)
        locthichnghicucbo_img = loc_thich_nghi_cuc_bo(gray,ksize=5,phuong_sai_nhieu=0.15)
        locthichnghicucbo_img_color = cv2.cvtColor(locthichnghicucbo_img, cv2.COLOR_GRAY2BGR)
        state["proc"] = locthichnghicucbo_img_color
        #Đẩy lên UI
        img_bytes = cv2_to_bytes(state["proc"])
        img_control.src_base64 = base64.b64encode(img_bytes).decode()
        page.update()
    
    def Frequency_LowpassFilter(img, D0=50):
        M, N = img.shape
        P, Q = 2 * M, 2 * N

        # Bước 1: mở rộng và dịch tâm
        f_xy_p = np.zeros((P, Q))
        f_xy_p[:M, :N] = img
        f_shift = f_xy_p * np.fromfunction(lambda x, y: (-1) ** (x + y), (P, Q))

        # Bước 2: DFT
        F_uv = np.fft.fft2(f_shift)

        # Bước 3: Tạo bộ lọc H(u,v)
        u = np.arange(P)
        v = np.arange(Q)
        U, V = np.meshgrid(u, v, indexing='ij')
        D = np.sqrt((U - P // 2)**2 + (V - Q // 2)**2)
        H = np.where(D <= D0, 1, 0)

        # Bước 4: Nhân F_uv với H
        G_uv = F_uv * H

        # Bước 5: IDFT và dịch lại tâm
        g_shift = np.fft.ifft2(G_uv).real
        g_xy_p = g_shift * np.fromfunction(lambda x, y: (-1) ** (x + y), (P, Q))

        # Bước 6: Cắt lại ảnh
        g_xy = g_xy_p[:M, :N]

        return g_xy.astype(np.float32)


    def frequency_lowpassfilter(e):
        if state["proc"] is None: return
        gray = cv2.cvtColor(state["proc"], cv2.COLOR_BGR2GRAY)
        frequency_lowpassfilter_img = Frequency_LowpassFilter(gray,D0=50)
        frequency_lowpassfilter_color = cv2.cvtColor(frequency_lowpassfilter_img, cv2.COLOR_GRAY2BGR)
        state["proc"] = frequency_lowpassfilter_color
        img_bytes = cv2_to_bytes(state["proc"])
        img_control.src_base64 = base64.b64encode(img_bytes).decode()
        page.update()

    def PhatHienKhuonMat(e):
        if state["proc"] is None: return
        gray = cv2.cvtColor(state["proc"], cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.05, 6)
        img = state["proc"].copy()
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 5)
        state["proc"] = img
        b = cv2_to_bytes(img)
        img_control.src_base64 = base64.b64encode(b).decode()
        page.update()

    def phathiencanhbien(img):
        def Convolution2D(img,kernel):
            m, n = img.shape
            img_new = np.zeros([m, n], dtype=np.float32)
            for i in range(1, m-1):
                for j in range(1, n-1):
                    temp = img[i-1, j-1] * kernel[0, 0] + \
                        img[i, j-1] * kernel[0, 1] + \
                        img[i+1, j-1] * kernel[0, 2] + \
                        img[i-1, j] * kernel[1, 0] + \
                        img[i, j] * kernel[1, 1] + \
                        img[i+1, j] * kernel[1, 2] + \
                        img[i-1, j+1] * kernel[2, 0] + \
                        img[i, j+1] * kernel[2, 1] + \
                        img[i+1, j+1] * kernel[2, 2]
                    img_new[i, j] = temp
            return img_new

        def Robertcross1(img):
            kernel = np.array([[0, 0, 0], 
                            [0, -1, 0], 
                            [0, 0, 1]], dtype=np.float32)
            return Convolution2D(img, kernel)

        def Robertcross2(img):
            kernel = np.array([[0, 0, 0], 
                            [0, 0, -1], 
                            [0, 1, 0]], dtype=np.float32)
            return Convolution2D(img, kernel)

    # gọi 2 hàm Robert cross
        grad1 = Robertcross1(img)
        grad2 = Robertcross2(img)

    # tính magnitude gradient
        grad = np.sqrt(grad1**2 + grad2**2)
        grad = np.clip(grad, 0, 255).astype(np.uint8)

        return grad
    def PhatHienCanhBien(e):
        if state["proc"] is None: return
        gray = cv2.cvtColor(state["proc"], cv2.COLOR_BGR2GRAY)
        canhbien_img = phathiencanhbien(gray)
        canhbien_color = cv2.cvtColor(canhbien_img, cv2.COLOR_GRAY2BGR)
        state["proc"] = canhbien_color
        img_bytes = cv2_to_bytes(state["proc"])
        img_control.src_base64 = base64.b64encode(img_bytes).decode()
        page.update()
    def reset(e):
        if state["orig"] is None: return
        state["proc"] = state["orig"].copy()
        b = cv2_to_bytes(state["proc"])
        img_control.src_base64 = base64.b64encode(b).decode()
        page.update()

    # Layout
    page.add(
    ft.Container(
        content=ft.Row(
            [
                # Cột 1: Tải ảnh + khung ảnh
                ft.Column(
                    [
                        ft.Text("ỨNG DỤNG XỬ LÝ ẢNH", size=26, weight="bold", text_align="center"),
                        ft.Row(
                            [
                                ft.ElevatedButton("Tải ảnh", icon=ft.Icons.UPLOAD_FILE,
                                    on_click=lambda e: file_picker.pick_files(
                                        allow_multiple=False, allowed_extensions=["jpg", "jpeg", "png", "tif"]
                                    )),
                                ft.ElevatedButton(" Khôi phục", icon=ft.Icons.REFRESH, on_click=reset),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=20
                        ),

                        ft.Container(
                            img_control,
                            alignment=ft.alignment.center,
                            padding=10,
                            bgcolor=ft.Colors.GREY_200,
                            border_radius=10,
                            margin=10,
                            height=400,
                            width=450,
                        ),
                    ],
                    spacing=20,
                    width=500,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),

                # Cột 2: Các chức năng lọc
                ft.Column(
                    [
                        ft.Text(" Bộ lọc trong không gian", size=20, weight="bold"),
                        ft.Row(
                            [
                                ft.ElevatedButton("Trung bình số học", on_click=Mean_Filter),
                                ft.ElevatedButton("Trung vị (Median)", on_click=Median_Filter),
                                ft.ElevatedButton("Thống kê Midpoint", on_click=Loc_tktt_midpoint),
                                ft.ElevatedButton("Alpha Trimmed", on_click=loc_tk_alpha_trimmed),
                                ft.ElevatedButton("Trung vị thích nghi", on_click=Loc_Trung_Vi_Thich_Nghi),
                                ft.ElevatedButton("Thích nghi cục bộ", on_click=Loc_Thich_Nghi_Cuc_Bo),
                                ft.ElevatedButton("Ideal Lowpass Filter", on_click=frequency_lowpassfilter),
                                ft.ElevatedButton("Phát hiện khuôn mặt", on_click=PhatHienKhuonMat),
                                ft.ElevatedButton("Phát hiện cạnh biên", on_click=PhatHienCanhBien),
                            ],
                            spacing=10,
                            wrap=True,
                        ),
                    ],
                    spacing=20,
                    width=400,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=50,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        padding=20,
        margin=20,
        border_radius=15,
        bgcolor=ft.Colors.WHITE,
        shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.GREY_500, offset=ft.Offset(2, 2))
    )
)



if __name__ == "__main__":
    # Mở cửa sổ riêng (desktop-like)
    ft.app(target=main)
