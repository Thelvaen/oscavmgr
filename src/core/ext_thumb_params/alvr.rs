use std::{
    sync::{
        mpsc::{Receiver, SyncSender},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use alvr_common::{
    glam::{EulerRot, Quat},
    DeviceMotion, Pose, HAND_LEFT_PATH, HAND_RIGHT_PATH, HEAD_PATH,
};

pub(super) struct AlvrReceiver {
    sender: SyncSender<Box<AlvrTrackingData>>,
    receiver: Receiver<Box<AlvrTrackingData>>,
    last_received: Instant,
}

impl AlvrReceiver {
    pub fn new() -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel(8);
        Self {
            sender,
            receiver,
            last_received: Instant::now(),
        }
    }
}
