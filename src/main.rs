extern crate sdl2;
extern crate rustfft;
extern crate num;
use num::complex::Complex;
use std::f64::consts::PI;
use std::f64;

const DIMS: (usize, usize) = (400, 400);
struct Point {
    pos: (i32, i32),
}

fn radians_to_rgb(rad: f64) -> (f64, f64, f64) {
    //outputs values bounded from 0 to 1
    (
        ((rad - PI/2.0).sin()).powf(2.0),
        ((rad - (4.0*PI/3.0) - PI/2.0).sin()).powf(2.0),
        ((rad - (2.0*PI/3.0) - PI/2.0).sin()).powf(2.0)
    )
}


fn main() {
    // set up all the SDL2 nonsense
    let ctx = sdl2::init()
        .expect("Couldn't initalize SDL2.");
    let video_ctx = ctx.video()
        .expect("Couldn't initalize SDL2 video context.");
    let window = video_ctx.window("FFT Drawpad", DIMS.0 as u32, DIMS.1 as u32)
        .position_centered()
        .build()
        .expect("Couldn't build SDL2 window.");
    let mut canvas = window
        .into_canvas()
        .build()
        .expect("Couldn't build SDL2 window canvas.");
    let canvas_tex_creator = canvas
        .texture_creator();
    let mut streaming_tex = canvas_tex_creator
        .create_texture_streaming(sdl2::pixels::PixelFormatEnum::RGB24, DIMS.0 as u32, DIMS.1 as u32)
        .expect("Couldn't capture canvas for a streaming texture.");


    // the main loop
    let mut point_list: Vec<Point> = vec![];
    let mut need_to_fft = false;
    // set up the FFT stuff

    let mut fft_in_buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); DIMS.0  * DIMS.1];
    let mut fft_out_buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); DIMS.0  * DIMS.1];
    let mut fft_render_buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); DIMS.0  * DIMS.1];
    let mut conv_matrix: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); DIMS.0  * DIMS.1];
    for (index, value) in (&mut conv_matrix).into_iter().enumerate() {
        let x: f64 = (index % DIMS.0) as f64;
        let y: f64 = (index / DIMS.0) as f64;
        *value = Complex::new(
            (((x - (DIMS.0 as f64 / 2.0)).powi(2) + (y - (DIMS.1 as f64 / 2.0)).powi(2)).sqrt()) / (DIMS.0 as f64 / 2.0),
            0.0
        );
    }
    let mut plan = rustfft::FFTplanner::new(false);
    let mut plan_inverse = rustfft::FFTplanner::new(true);
    let fft = plan.plan_fft(DIMS.0 * DIMS.1);
    let fft_inverse = plan_inverse.plan_fft(DIMS.0 * DIMS.1);
    let mut conv_matrix_fft: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); DIMS.0 * DIMS.1];
    fft.process(&mut conv_matrix, &mut conv_matrix_fft);

    'main : loop {
        // take input
        for event in ctx.event_pump().expect("What the fuck?").poll_iter() {
            use sdl2::event::Event;
            match event {
                Event::MouseButtonDown{x, y, mouse_btn: sdl2::mouse::MouseButton::Left, ..} => {
                    point_list.push(Point {pos: (x, y)});
                    fft_in_buffer[(x + (y * DIMS.0 as i32)) as usize] = Complex::new(255.0, 0.0);
                    need_to_fft = true;
                }
                Event::Quit{..} => break 'main,
                _ => (),
            }
        }


        // PERFORM THE FFT ON WINDOW 0'S CONTENTS
        // first, transform the data into something that the dft crate can understand
        // that result is in fft_buffer -- which is already allocated
        if need_to_fft {
            need_to_fft = false;
            // first save input vector
            let fft_in_buffer_backup = fft_in_buffer.clone();
            // perform the DFT
            fft.process(&mut fft_in_buffer, &mut fft_out_buffer);
            // and restore the input vector
            fft_in_buffer = fft_in_buffer_backup;

            // next, convolute the output buffer
            for (index, value) in (&mut fft_out_buffer).into_iter().enumerate() {
                *value *= conv_matrix_fft[index];
            }

            // perform the inverse transform
            fft_inverse.process(&mut fft_out_buffer, &mut fft_render_buffer);
            // find the max value in the FFT out buffer
            let mut max_value: f64 = 0.0;
            for value in &fft_render_buffer {
                if value.norm().round() > max_value {
                    max_value = value.norm().round();
                }
            }


            // copy the fft_out_buffer WINDOW 1's canvas streaming texture
            streaming_tex.with_lock(None, |buffer: &mut [u8], _pitch: usize| {
                for (index, value) in (&fft_render_buffer).into_iter().enumerate() {
                    let amplitude: f64 = value.norm().floor(); // 0 to max_value
                    let brightness: f64 = 256.0 * amplitude / max_value; // 0 to 255
                    let rgb: (f64, f64, f64) = radians_to_rgb(value.arg()); // all elements are 0 to 1
                    buffer[index * 3    ] = (brightness * rgb.0).floor() as u8;
                    buffer[index * 3 + 1] = (brightness * rgb.1).floor() as u8;
                    buffer[index * 3 + 2] = (brightness * rgb.2).floor() as u8;
                }
            }).expect("What the fuck?");
        }
        // update canvases from texture
        canvas.clear();
        canvas.copy(&streaming_tex, None, None).expect("What the fuck?");
        for p in &point_list {
            // draw the points
            canvas.set_draw_color(sdl2::pixels::Color::RGB(255, 0, 0));
            canvas.fill_rect(sdl2::rect::Rect::new(p.pos.0, p.pos.1, 5, 5));
        }
        canvas.present();

    }
}
