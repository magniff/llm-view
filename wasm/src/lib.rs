//! The engine as a WebAssembly module, with a deliberately tiny ABI and no
//! binding generator. JavaScript (`viz/web/worker.js`) talks to it like this:
//!
//! 1. **Passing bytes in**: JS calls `alloc(n)`, copies `n` bytes into the
//!    returned spot of the module's linear memory, and passes `(ptr, n)` to
//!    an export, which takes ownership and frees it.
//! 2. **Getting bytes out**: every export returns a status (0 = ok, 1 = error)
//!    and leaves its JSON result, or error message, in an output buffer that JS
//!    reads via `out_ptr()` / `out_len()`.
//!
//! Loading happens in three steps so that the 269 MB weight file never has to
//! sit in wasm memory next to its 538 MB f32 copy:
//!
//! - `init(config, tokenizer, header)`: parse the small files and the safetensors
//!   header, and reply with the byte range of every tensor;
//! - `put_tensor(name, bytes)` once per tensor: decode bf16 → f32 and keep it;
//! - `finish(max_seq, chat_ctx)`: assemble the model and start a [`Session`].
//!
//! After that, `call(method, args)` is the whole API (see `engine::session`).

use std::cell::RefCell;
use std::collections::HashMap;

use engine::model::{Config, Model};
use engine::safetensors::{decode, parse_header, TensorInfo};
use engine::session::Session;
use engine::tensor::Tensor;
use engine::tokenizer::Tokenizer;
use serde_json::{json, Value};

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// Print a UTF-8 string to the browser console (used for panics).
    fn log_str(ptr: *const u8, len: usize);
}

enum State {
    Empty,
    Loading { cfg: Config, tok: Tokenizer, infos: HashMap<String, TensorInfo>, tensors: HashMap<String, Tensor>, name: String },
    Ready(Box<Session>),
}

thread_local! {
    static STATE: RefCell<State> = const { RefCell::new(State::Empty) };
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------- memory

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Take back a buffer JS filled after `alloc(len)`.
unsafe fn take(ptr: *mut u8, len: usize) -> Vec<u8> {
    // SAFETY: `ptr` came from `alloc(len)` and JS wrote exactly `len` bytes.
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

unsafe fn take_str(ptr: *mut u8, len: usize) -> String {
    // SAFETY: as for `take`.
    String::from_utf8_lossy(&unsafe { take(ptr, len) }).into_owned()
}

#[unsafe(no_mangle)]
pub extern "C" fn out_ptr() -> *const u8 {
    OUT.with(|o| o.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn out_len() -> usize {
    OUT.with(|o| o.borrow().len())
}

fn reply(r: Result<Value, String>) -> u32 {
    let (status, body) = match r {
        Ok(v) => (0, v.to_string()),
        Err(e) => (1, e),
    };
    OUT.with(|o| *o.borrow_mut() = body.into_bytes());
    status
}

// ---------------------------------------------------------------- loading

/// Called once by the JS glue at startup: route panics to the console.
#[unsafe(no_mangle)]
pub extern "C" fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = format!("engine panic: {info}");
        // SAFETY: a valid UTF-8 buffer for the duration of the call.
        unsafe { log_str(msg.as_ptr(), msg.len()) }
    }));
}

/// # Safety
/// All pointer/length pairs must come from `alloc` and be fully written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init(
    cfg_ptr: *mut u8,
    cfg_len: usize,
    tok_ptr: *mut u8,
    tok_len: usize,
    hdr_ptr: *mut u8,
    hdr_len: usize,
    name_ptr: *mut u8,
    name_len: usize,
) -> u32 {
    // SAFETY: forwarded from the caller's contract.
    let (cfg, tok, hdr, name) = unsafe {
        (take_str(cfg_ptr, cfg_len), take_str(tok_ptr, tok_len), take(hdr_ptr, hdr_len), take_str(name_ptr, name_len))
    };
    reply((|| {
        let cfg = Config::from_json_str(&cfg).map_err(|e| e.to_string())?;
        let tok = Tokenizer::from_json_str(&tok).map_err(|e| e.to_string())?;
        let infos = parse_header(&hdr).map_err(|e| e.to_string())?;
        let mut list: Vec<(&String, &TensorInfo)> = infos.iter().collect();
        list.sort_by_key(|(_, i)| i.offsets.0);
        let ranges: Vec<Value> = list.iter().map(|(n, i)| json!([n, i.offsets.0, i.offsets.1])).collect();
        let out = json!({"tensors": ranges});
        STATE.with(|s| *s.borrow_mut() = State::Loading { cfg, tok, infos, tensors: HashMap::new(), name });
        Ok(out)
    })())
}

/// # Safety
/// Both pointer/length pairs must come from `alloc` and be fully written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn put_tensor(name_ptr: *mut u8, name_len: usize, data_ptr: *mut u8, data_len: usize) -> u32 {
    // SAFETY: forwarded from the caller's contract.
    let (name, data) = unsafe { (take_str(name_ptr, name_len), take(data_ptr, data_len)) };
    reply(STATE.with(|s| match &mut *s.borrow_mut() {
        State::Loading { infos, tensors, .. } => {
            let info = infos.get(&name).ok_or_else(|| format!("unknown tensor {name}"))?;
            let t = decode(&name, info, &data).map_err(|e| e.to_string())?;
            tensors.insert(name, t);
            Ok(Value::Null)
        }
        _ => Err("put_tensor: call init first".to_string()),
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn finish(max_seq: usize, chat_ctx: usize) -> u32 {
    reply(STATE.with(|s| {
        let st = std::mem::replace(&mut *s.borrow_mut(), State::Empty);
        let State::Loading { cfg, tok, mut tensors, name, .. } = st else {
            return Err("finish: call init first".to_string());
        };
        let model = Model::from_loader(cfg, |n| tensors.remove(n).ok_or_else(|| format!("missing tensor {n}").into()))
            .map_err(|e| e.to_string())?;
        let session = Session::new(model, tok, &name, max_seq, chat_ctx)?;
        *s.borrow_mut() = State::Ready(Box::new(session));
        Ok(Value::Null)
    }))
}

// ---------------------------------------------------------------- the API

/// # Safety
/// Both pointer/length pairs must come from `alloc` and be fully written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn call(method_ptr: *mut u8, method_len: usize, args_ptr: *mut u8, args_len: usize) -> u32 {
    // SAFETY: forwarded from the caller's contract.
    let (method, args) = unsafe { (take_str(method_ptr, method_len), take_str(args_ptr, args_len)) };
    reply(STATE.with(|s| match &mut *s.borrow_mut() {
        State::Ready(session) => {
            let args: Value = if args.is_empty() { Value::Null } else { serde_json::from_str(&args).map_err(|e| e.to_string())? };
            session.call(&method, &args)
        }
        _ => Err("model not loaded yet".to_string()),
    }))
}
