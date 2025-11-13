use std::fs::File;
use std::time::Instant;

use rosc::{OscBundle, OscType};
use serde::{Deserialize, Serialize};

use super::bundle::AvatarBundle;
use super::folders::CONFIG_DIR;
use super::AvatarParameters;

use openxr as xr;

const FILE_NAME: &str = "ExtThumbParams.json";

const CONTROLLER_TYPE: &str = "ControllerType";
const LEFT_THUMB: &str = "LeftThumb";
const RIGHT_THUMB: &str = "RightThumb";
const LEFT_TRIGGER: &str = "LeftTrigger";
const RIGHT_TRIGGER: &str = "RightTrigger";

#[derive(Serialize, Deserialize, Default)]
pub struct ExtThumbParams {
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    path: String,
}

impl ExtThumbParams {
    pub fn new() -> ExtThumbParams {
        let path = format!("{}/{}", CONFIG_DIR.as_ref(), FILE_NAME);

        let mut me = File::open(&path)
            .ok()
            .and_then(|file| serde_json::from_reader(file).ok())
            .unwrap_or_else(|| Some(ExtThumbParams::default()))
            .unwrap();

        me.path = path;

        me
    }

    fn save(&mut self) {
        log::info!("Saving ExtThumbParams to {}", &self.path);
        File::create(&self.path)
            .ok()
            .and_then(|file| serde_json::to_writer(file, self).ok());
    }

    pub fn step(&mut self, bundle: &mut OscBundle) {
        let controller_type = get_controller_type(); // i32
        let left_thumb = get_left_thumb(); // i32
        let right_thumb = get_right_thumb(); // i32
        let left_trigger = get_left_trigger(); // f32
        let right_trigger = get_right_trigger(); // f32

        bundle.send_parameter(CONTROLLER_TYPE, OscType::Int(controller_type));
        bundle.send_parameter(LEFT_THUMB, OscType::Int(left_thumb));
        bundle.send_parameter(RIGHT_THUMB, OscType::Int(right_thumb));
        bundle.send_parameter(LEFT_TRIGGER, OscType::Float(left_trigger));
        bundle.send_parameter(RIGHT_TRIGGER, OscType::Float(right_trigger));
    }
}
