use image::codecs::jpeg;
use isahc::ReadResponseExt;
use memmap2::MmapOptions;
use std::fs::OpenOptions;
use std::io::Cursor;

use std::fs::File;

use image::ImageReader;

#[cfg(feature = "mpi")]
use mpi::traits::Communicator;

pub const ZOOM_LV: u32 = 3;
pub const OUTPUT_BIN_PATH: &str = "./result.bin";
pub const OUTPUT_JPG_PATH: &str = "./result.jpg";

const TILE_SIZE: u64 = 256;
const NUM_OF_PIXELS: u64 = TILE_SIZE * TILE_SIZE;

#[cfg(feature = "mpi")]
fn mpi_init() -> (usize, usize) {
    let universe = mpi::initialize().expect("Faild to initialize MPI");
    let world = universe.world();
    let size = world.size() as u64;
    let rank = world.rank() as u64;
    (size, rank)
}

// MPIが無効の場合は1プロセスで実行
#[cfg(not(feature = "mpi"))]
fn mpi_init() -> (u64, u64) {
    (1, 0)
}

// 画像を取得して標高データを返す
fn fetch(x: u64, y: u64) -> anyhow::Result<Vec<Vec<f32>>> {
    let url = format!("https://tiles.gsj.jp/tiles/elev/land/{ZOOM_LV}/{y}/{x}.png");
    // いちいちクライアントを作るので遅い
    let mut res = isahc::get(&url).expect("Failed to get");

    if res.status().is_success() {
        let bytes = res.bytes().unwrap();

        let img = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();

        // 画像のRGB値を標高データに変換
        let altitude = img
            .pixels()
            .map(|pixel| {
                let r = pixel[0] as f64;
                let g = pixel[1] as f64;
                let b = pixel[2] as f64;

                let x = 2_f64.powi(16) * r + 2_f64.powi(8) * g + b;
                let u = 0.01;

                if x < 2_f64.powi(23) {
                    (x * u) as f32
                } else if x > 2_f64.powi(23) {
                    ((x - 2_f64.powi(24)) * u) as f32
                } else {
                    f32::MIN
                }
            })
            .collect::<Vec<_>>()
            .chunks(TILE_SIZE as usize)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(altitude)
    } else {
        // タイル全体が存在しない場合
        Ok(vec![vec![f32::MIN; TILE_SIZE as usize]; TILE_SIZE as usize])
    }
}

fn main() -> anyhow::Result<()> {
    let (size, rank) = mpi_init();

    // スレッド数は2の累乗でなければならない
    assert_eq!(
        size & (size - 1),
        0,
        "The number of processes must be a power of 2"
    );

    // スレッド数は1辺のタイル数よりも少なくなければならない
    assert!(
        size <= 2_u64.pow(ZOOM_LV),
        "The number of processes must be less than or equal to the number of tiles on one side"
    );

    // ランク0のプロセスにファイルを作らせる
    if rank == 0 {
        let file = File::create(OUTPUT_BIN_PATH)?;
        let file_size = NUM_OF_PIXELS * 2_u64.pow(ZOOM_LV).pow(2) * size_of::<f32>() as u64;
        file.set_len(file_size)?;
    }

    // 1プロセスあたりの処理量（タイル群高さ）
    let height = 2_u64.pow(ZOOM_LV) / size;
    let start_h = rank * height;
    let end_h = (rank + 1) * height;

    // タイル群の幅
    let weidth = 2_u64.pow(ZOOM_LV);

    let big_tile = {
        let mut field =
            vec![vec![f32::MIN; (weidth * TILE_SIZE) as usize]; (height * TILE_SIZE) as usize];

        // 256x256の標高タイルの配列
        (start_h..end_h).into_iter().for_each(|tile_y| {
            (0..weidth).into_iter().for_each(|tile_x| {
                if let Ok(tile) = fetch(tile_x, tile_y) {
                    tile.into_iter().enumerate().for_each(|(pixel_y, row)| {
                        row.into_iter().enumerate().for_each(|(pixel_x, altitude)| {
                            let y = (tile_y * TILE_SIZE) as usize + pixel_y;
                            let x = (tile_x * TILE_SIZE) as usize + pixel_x;
                            field[y as usize][x as usize] = altitude;
                        })
                    })
                }
            })
        });

        field
            .into_iter()
            .flat_map(|row| row.into_iter().flat_map(|f| f.to_be_bytes()))
            .collect::<Vec<_>>()
    };

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(OUTPUT_BIN_PATH)
        .expect("Failed to open file");

    let mut mmap = unsafe {
        let start = NUM_OF_PIXELS * (start_h * weidth) * size_of::<f32>() as u64;
        MmapOptions::new()
            .offset(start)
            .len((NUM_OF_PIXELS * (end_h - start_h) * weidth) as usize * size_of::<f32>())
            .map_mut(&file)
    }
    .expect("Failed to mmap");
    let start_ptr = mmap.as_mut_ptr();

    unsafe {
        big_tile.iter().enumerate().for_each(|(offset, &byte)| {
            let ptr = start_ptr.add(offset);
            ptr.write(byte);
        });
    }

    let pixels = big_tile
        .chunks(4)
        .map(|chunk| {
            let float = f32::from_be_bytes(chunk.try_into().unwrap()) as f64;

            // 0から8848(エベレスト)までの値に正規化
            let normalized = (float / 8848.0) * 255.0;
            normalized as u8
        })
        .collect::<Vec<_>>();

    let jpg = File::create(OUTPUT_JPG_PATH)?;
    let mut jpg_encoder = jpeg::JpegEncoder::new(jpg);
    let image_size = TILE_SIZE * 2_u64.pow(ZOOM_LV);
    jpg_encoder
        .encode(
            pixels.as_slice(),
            image_size as u32,
            image_size as u32,
            image::ExtendedColorType::L8,
        )
        .unwrap();

    Ok(())
}
