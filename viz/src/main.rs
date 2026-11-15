//! Native server for the visualizer (for local development; the page can
//! also run the engine entirely in the browser via WebAssembly; see `viz/web/`).
//!
//!   cargo run --release -p viz                 # http://127.0.0.1:7860/  (?native: use this server)
//!   cargo run --release -p viz -- --port 8000
//!   cargo run --release -p viz -- --static     # files only: no model loaded here
//!
//! The visualizer uses SmolLM2-135M-Instruct.
//!
//! It also serves the repo's `models/` folder at `/models/`, which the
//! in-browser engine uses when the page is opened on localhost, so testing
//! doesn't re-download weights from Hugging Face.
//!
//! All model work goes through [`engine::session::Session::call`]; this file only
//! puts it behind `POST /api/call/{method}` and serves `viz/web/` from disk.

use std::sync::{Arc, Mutex};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use engine::download;
use engine::model::Model;
use engine::session::Session;
use engine::tokenizer::Tokenizer;
use serde_json::{json, Value};
use tower_http::services::ServeDir;

type Shared = Arc<Mutex<Session>>;

async fn call(State(st): State<Shared>, Path(method): Path<String>, body: Option<Json<Value>>) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let args = body.map(|Json(v)| v).unwrap_or(Value::Null);
    // Model work is CPU-heavy: keep it off the async executor.
    let r = tokio::task::spawn_blocking(move || {
        let mut s = st.lock().unwrap_or_else(|e| e.into_inner());
        s.call(&method, &args)
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r);
    r.map(Json).map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))
}

fn main() -> engine::Result<()> {
    let (mut port, mut stat) = (7860u16, false);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--static" => stat = true,
            "--port" => port = it.next().ok_or("--port needs a value")?.parse()?,
            _ => return Err(format!("unknown argument {a} (use --port N, --static)").into()),
        }
    }
    let spec = &download::SMOLLM2_135M_INSTRUCT;
    let models = concat!(env!("CARGO_MANIFEST_DIR"), "/../models");
    let dir = format!("{models}/{}", spec.dir);
    // The in-browser engine fetches from /models/ on localhost, so make sure it's there.
    download::ensure_model(std::path::Path::new(&dir), spec)?;

    let web = concat!(env!("CARGO_MANIFEST_DIR"), "/web");
    let mut app = Router::new();
    if !stat {
        let model = Model::load(&dir)?;
        let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json"))?;
        let session = Session::new(model, tok, spec.dir, 1024, 2048)?;
        app = app.route("/api/call/{method}", post(call).with_state(Arc::new(Mutex::new(session))));
    }
    let app = app.nest_service("/models", ServeDir::new(models)).fallback_service(ServeDir::new(web));

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let addr = format!("127.0.0.1:{port}");
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        if stat {
            eprintln!("serving files only · open http://{addr}/ (engine runs in the browser, weights from /models/)");
        } else {
            eprintln!("{} loaded · open http://{addr}/ (in-browser engine) or http://{addr}/?native (this server)", spec.dir);
        }
        axum::serve(listener, app).await?;
        Ok(())
    })
}
