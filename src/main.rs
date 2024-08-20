use std::io::{BufReader,BufWriter,Write ,Cursor, SeekFrom, Seek};
use futures::future::join_all;

use std::fs::File;

use image::ImageReader;

#[cfg(feature = "mpi")]
use mpi::traits::Communicator;

const ZOOM_LV: u32 = 2;
const OUTPUT_FILE: &str = "./result.bin";
const WRITER_CAPACITY: usize = size_of::<f32>() * 1024;

const NUM_OF_PIXELS: u64 = 256 * 256;

#[cfg(feature = "mpi")]
fn mpi_init() -> (usize, usize) {
    let universe = mpi::initialize().expect("Faild to initialize MPI");
    let world = universe.world();
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    (size, rank)
}

// MPIが無効の場合は1プロセスで実行
#[cfg(not(feature = "mpi"))]
fn mpi_init() -> (usize, usize) {
    (1, 0)
}

// 画像を取得して標高データを返す
async fn fetch(x: usize, y: usize) -> anyhow::Result<Vec<f32>> {
    let url = format!("https://tiles.gsj.jp/tiles/elev/land/{ZOOM_LV}/{y}/{x}.png");
    // いちいちクライアントを作るので遅い
    let res = reqwest::get(&url).await?.bytes().await?;

    let cursor = Cursor::new(res);
    let reader = BufReader::new(cursor);
    let png = ImageReader::new(reader).decode()?;

    // 画像のRGB値を標高データに変換
    let altitude= png.into_rgb8().pixels().map(|pixel| {
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
    }).collect();

    Ok(altitude)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (size, rank) = mpi_init();

    // スレッド数は2の累乗でなければならない
    assert_eq!(
        size & (size - 1),
        0,
        "The number of processes must be a power of 2"
    );

    // ランク0のプロセスにファイルを作らせる
    if rank == 0 {
        let file = File::create(OUTPUT_FILE)?;
        let file_size = NUM_OF_PIXELS * (2_u64.pow(ZOOM_LV)*size_of::<f32>() as u64).pow(2);
        file.set_len(file_size)?;
    }


    // 1プロセスあたりの処理量（タイル群高さ）
    let height = 2_usize.pow(ZOOM_LV) / size;
    let start_h = rank * height;
    let end_h = (rank + 1) * height;

    // タイル群の幅
    let weidth = 2_usize.pow(ZOOM_LV);
    
    let works = (start_h..end_h).flat_map(|y|
        (0..weidth).map(move |x| fetch(x, y))
    );

    let result_bytes = join_all(works).await.into_iter().flatten().flatten().flat_map(|f|f.to_be_bytes());
    
    let mut file = File::open(OUTPUT_FILE)?;
    let offset = NUM_OF_PIXELS * (start_h * weidth * size_of::<f32>()) as u64;
    file.seek(SeekFrom::Start(offset))?;
    
    let mut writer = BufWriter::with_capacity(WRITER_CAPACITY, file);
    
    writer.write_all(&result_bytes.collect::<Vec<u8>>())?;

    Ok(())
}
