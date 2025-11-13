use openxr as xr;
use strum::EnumCount;

pub struct OpenXrReceiver {
    state: Option<XrState>,
    last_attempt: Instant,
}

impl OpenXrReceiver {
    pub fn new() -> Self {
        Self {
            state: None,
            last_attempt: Instant::now(),
        }
    }

    fn try_init(&mut self) {
        self.state = XrState::new().map_err(|e| log::error!("XR: {}", e)).ok();
        self.last_attempt = Instant::now();
    }
}
