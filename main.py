from functools import lru_cache
from pathlib import Path
import shutil
import pyautogui
from loguru import logger
from PIL import Image
import typing
# from ocrmac import ocrmac。
import img2pdf
import numpy as np

class Point(typing.NamedTuple):
    x: int
    y: int

class Rect(typing.NamedTuple):
    x: int
    y: int
    width: int
    height: int


def get_book_rect() -> Rect:
    print("请将鼠标移动到书本的左上角, 然后回车")
    input()
    pos1 = pyautogui.position()
    print(f"书本的左上角坐标: {pos1}")

    print("请将鼠标移动到书本的右下角, 然后回车")
    input()
    pos2 = pyautogui.position()
    print(f"书本的右下角坐标: {pos2}")
    return Rect(pos1[0], pos1[1], pos2[0] - pos1[0], pos2[1] - pos1[1])

def get_next_page_button_point() -> Point:
    print("请将鼠标移动到下一页按钮, 然后回车")
    input()
    pos = pyautogui.position()
    print(f"下一页按钮坐标: {pos}")
    return Point(pos[0], pos[1])


def get_white_percent(image: Image.Image) -> float:
    white_pixels = 0
    for pixel in image.getdata():
        # Allow approximately 10 points of RGB error for white detection
        if all(value >= 245 for value in pixel[:3]):  # Check RGB values are close to white (255)
            white_pixels += 1
    return white_pixels / (image.width * image.height)


def compress_save(image: Image.Image, path: Path):
    """Compress and save image with optimizations for text content to minimize file size.
    
    This function performs several optimizations:
    1. Convert to grayscale
    2. Apply contrast enhancement to make text more distinct
    3. Apply binary thresholding (black and white only)
    4. Save with high compression settings
    """
    # Step 1: Convert to grayscale
    img_gray = image.convert("L")
    
    # Step 2: Enhance contrast to make text more distinct
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img_gray)
    img_enhanced = enhancer.enhance(2.0)  # Increase contrast
    
    # Step 3: Apply binary thresholding (only black and white pixels)
    # Adjust threshold value (128) if needed for better results
    img_binary = img_enhanced.point(lambda x: 0 if x < 128 else 255, '1')
    
    # Step 4: Save with optimal format and settings
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        # For JPEG, use high quality for text
        img_gray.save(path, optimize=True, quality=30)
    elif path.suffix.lower() == '.png':
        # For PNG, use maximum compression
        img_binary.save(path, optimize=True, compress_level=9)
    elif path.suffix.lower() == '.webp':
        # WebP often provides the best compression for text
        img_gray.save(path, format='WEBP', lossless=False, quality=30)
    else:
        # Default to binary image with PNG compression
        img_binary.save(path, optimize=True)
    

@lru_cache
def get_scale() -> float:
    im = pyautogui.screenshot()
    return im.size[0] / pyautogui.size()[0]

last_screenshot: typing.Optional[Image.Image] = None

def calculate_image_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Calculate similarity between two images using Mean Squared Error (MSE).
    Returns a value between 0 and 1, where 1 means identical images."""
    if img1 is None or img2 is None:
        return 0.0
    
    # Ensure images are the same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    
    # Convert images to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate mean squared error
    mse = float(np.mean(((arr1 / 255.0) - (arr2 / 255.0)) ** 2))
    
    # Convert MSE to similarity score (1 - normalized MSE)
    # The smaller the MSE, the more similar the images
    similarity = 1.0 - min(1.0, mse)
    
    return similarity

class SameScreenshotException(Exception):
    pass


def get_book_screenshot(rect: Rect) -> Image.Image:
    global last_screenshot
    for _ in range(5):
        img = pyautogui.screenshot() #region=(rect.x, rect.y, rect.width, rect.height))
        scale = get_scale()
        crop_rect = (rect.x*scale, rect.y*scale, (rect.x + rect.width)*scale, (rect.y + rect.height)*scale)
        img = img.crop(crop_rect).convert("RGB")
        # 获取图片中白色占比，如果太高就sleep一下重新截图
        percent = get_white_percent(img)
        if percent > 0.999:
            print(f"图片中白色占比: {percent}")
            pyautogui.sleep(1)
            continue
        
        if last_screenshot is not None:
            similarity = calculate_image_similarity(img, last_screenshot)
            print(f"与上一页相似度: {similarity:.4f}")
            if similarity > 0.999:
                if '已读完' in image_to_text(img):
                    raise StopIteration()
                print("检测到相似页面，可能截图太快了")
                continue
        
        last_screenshot = img
        return img
    raise Exception("无法获取书本截图")

def image_to_text(image: Image.Image) -> str:
    from ocrmac import ocrmac
    annotations = ocrmac.OCR(image, language_preference=['zh-Hans', 'en-US']).recognize()
    text = []
    for item in annotations:
        # print(item)
        text.append(item[0])
    return "\n".join(text)


def merge_pdf(imgdir: Path):
    # merge into pdf
    print("Merging images into PDF...")
    image_files = sorted(imgdir.glob("book_*.jpg"))
    pdf_path = Path("book_capture.pdf")
    
    # Convert images to PDF bytes
    pdf_bytes = img2pdf.convert([str(img_path) for img_path in image_files])
    
    # Write PDF bytes to file
    if pdf_bytes is not None:
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"PDF created successfully: {pdf_path.absolute()}")
    else:
        print("Failed to create PDF: No bytes generated")
    
def main():
    book_rect = get_book_rect()
    next_page_button_point = get_next_page_button_point()



    print(f"下一页按钮坐标: {next_page_button_point}")
    pyautogui.click(next_page_button_point.x, next_page_button_point.y)
    # return

    # with open("book_screenshot.pdf", "wb") as f:
    dst = Path('capimgs')
    # clean dst
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(exist_ok=True)

    page_index = 0
    while True:
        try:
            print(f"正在捕获第 {page_index} 页...")
            book_screenshot = get_book_screenshot(book_rect)
            output_path = Path(f"capimgs/book_{page_index:03d}.jpg")
            compress_save(book_screenshot, output_path)
            print(f"保存图片: {output_path} (大小: {output_path.stat().st_size / 1024:.1f} KB)")
            pyautogui.click(next_page_button_point.x, next_page_button_point.y)
            pyautogui.sleep(.1)
        except StopIteration:
            print("已读完")
            break
        except Exception as e:
            print(f"捕获出错: {e}")
            break
        finally:
            page_index += 1
    
    merge_pdf(dst)


if __name__ == '__main__':
    # image = Image.open('capimgs/book_002.jpg')
    # compress_save(image, Path('capimgs/book_000-bak.png'))
    main()