import asyncio
import os

import aiofiles
import aiohttp

# Đường dẫn file txt chứa dữ liệu
input_file = "/media/DATA/Fashion-Recommendation-System-/fashion-iq-metadata/image_url/merged.txt"  # thay bằng file của bạn
output_folder = "/media/DATA/Fashion-Recommendation-System-/data/images"

os.makedirs(output_folder, exist_ok=True)


# Hàm tải 1 ảnh
async def download_image(session, code, url, sem):
    async with sem:  # giới hạn số request đồng thời
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    file_path = os.path.join(output_folder, f"{code}.jpg")
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await resp.read())
                    return None
                else:
                    return f"Lỗi {code}: HTTP {resp.status}"
        except Exception as e:
            return f"Lỗi {code}: {e}"


async def main():
    # Đọc dữ liệu
    tasks_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tasks_data.append((parts[0], parts[1]))

    connector = aiohttp.TCPConnector(limit=0)  # không giới hạn kết nối TCP
    sem = asyncio.Semaphore(
        100
    )  # giới hạn đồng thời 100 request để tránh nghẽn/bị chặn

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_image(session, code, url, sem) for code, url in tasks_data]
        for future in asyncio.as_completed(tasks):
            result = await future
            if result:  # chỉ in nếu có lỗi
                print(result)


if __name__ == "__main__":
    asyncio.run(main())
