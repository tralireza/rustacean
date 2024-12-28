use graphics::types::Vec2d;
use std::alloc::{GlobalAlloc, Layout, System};

/*
#[global_allocator]
static ALLOCATOR: LoggerAllocator = LoggerAllocator;
*/

struct LoggerAllocator;

unsafe impl GlobalAlloc for LoggerAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        use std::time::Instant;

        let start = Instant::now();
        let ptr = System.alloc(layout);
        let end = Instant::now();
        let time_taken = end - start;
        let bytes_requested = layout.size();

        eprintln!("{}\t{}", bytes_requested, time_taken.as_nanos());

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

struct World {
    turn: u64,
    particles: Vec<Box<Particle>>,
    height: f64,
    width: f64,
    rng: rand::rngs::ThreadRng,
}

impl World {
    fn new(width: f64, height: f64) -> Self {
        World {
            turn: 0,
            particles: Vec::<Box<Particle>>::new(),
            height: height,
            width: width,
            rng: rand::thread_rng(),
        }
    }

    fn add(&mut self, n: i32) {
        for _ in 0..n.abs() {
            self.particles.push(Box::new(Particle::new(&self)))
        }
    }

    fn remove(&mut self, n: i32) {
        for _ in 0..n.abs() {
            let mut dp = None;

            for (i, p) in self.particles.iter().enumerate() {
                if p.color[3] < 0.02 {
                    dp = Some(i);
                }
                break;
            }

            if let Some(i) = dp {
                self.particles.remove(i);
            } else {
                self.particles.remove(0);
            }
        }
    }

    fn update(&mut self) {
        use rand::Rng;

        let n = self.rng.gen_range(-3..=3);
        if n > 0 {
            self.add(n);
        } else {
            self.remove(n);
        }

        self.particles.shrink_to_fit();
        for p in &mut self.particles {
            p.update();
        }
        self.turn += 1;
    }
}

struct Particle {
    height: f64,
    width: f64,
    position: Vec2d<f64>,
    velocity: Vec2d<f64>,
    acceleration: Vec2d<f64>,
    color: [f32; 4],
}

impl Particle {
    fn new(world: &World) -> Self {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let (x, y) = (rng.gen_range(0.0..=world.width), world.height);
        let (xv, yv) = (0.0, rng.gen_range(-2.0..0.0));
        let (xa, ya) = (0.0, rng.gen_range(0.0..0.15));

        Particle {
            height: 4.0,
            width: 4.0,
            position: [x, y].into(),
            velocity: [xv, yv].into(),
            acceleration: [xa, ya].into(),
            color: [1.0, 1.0, 1.0, 0.99],
        }
    }

    fn update(&mut self) {
        use graphics::math::add;

        self.velocity = add(self.velocity, self.acceleration);
        self.position = add(self.position, self.velocity);
        self.acceleration = graphics::math::mul_scalar(self.acceleration, 0.7);
        self.color[3] *= 0.995;
    }
}

fn main() {
    let (width, height) = (1280.0, 800.0);

    let mut window: piston_window::PistonWindow =
        piston_window::WindowSettings::new("Graphics!", [width, height])
            .exit_on_esc(true)
            .build()
            .expect("Can't create a Window!");

    let mut world = World::new(width, height);
    world.add(1000);

    while let Some(event) = window.next() {
        world.update();

        window.draw_2d(&event, |ctx, renderer, _| {
            piston_window::clear([0.15, 0.17, 0.17, 0.9], renderer);

            for p in &mut world.particles {
                let size = [p.position[0], p.position[1], p.width, p.height];
                piston_window::rectangle(p.color, size, ctx.transform, renderer);
            }
        });
    }
}
