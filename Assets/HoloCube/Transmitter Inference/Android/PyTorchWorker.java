package com.holocube.pytorch;

import android.content.Context;
import android.util.Log;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Owns one background thread. Unity only submits images and polls for results. */
public final class PyTorchWorker {
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Future<PyObject> startup;
    private Future<float[]> pending;
    private boolean discarded;
    private boolean closed;
    private boolean reportedFrame;
    private boolean reportedOrientationFrame;
    private volatile boolean orientationReady;
    private volatile String orientationError;

    public PyTorchWorker(Context context, int threads) {
        Context app = context.getApplicationContext();
        startup = executor.submit(() -> {
            synchronized (Python.class) {
                if (!Python.isStarted()) Python.start(new AndroidPlatform(app));
            }
            PyObject runtime = Python.getInstance().getModule("holocube_runtime");
            Log.i("HoloCube", runtime.callAttr("initialize", threads).toString());
            Log.i("HoloCube", "HOLOCUBE_PT_READY: Transmitter Recognition Model loaded on device; class 0: Transmitter");
            try {
                Log.i("HoloCube", runtime.callAttr("initialize_orientation").toString());
                orientationReady = true;
                Log.i("HoloCube", "HOLOCUBE_ORIENTATION_READY: Orientation Detection Model loaded on device; classes North to Northwest");
            } catch (Exception error) {
                // The transmitter passed its own check and stays usable without this model.
                orientationError = String.valueOf(error.getMessage());
                Log.e("HoloCube", "HOLOCUBE_ORIENTATION_UNAVAILABLE: " + orientationError, error);
            }
            return runtime;
        });
    }

    public synchronized boolean isReady() {
        if (closed) throw new IllegalStateException("PyTorch worker is closed");
        if (!startup.isDone()) return false;
        get(startup); // Surface initialization errors to the headset UI.
        return true;
    }

    public synchronized boolean isIdle() {
        if (!isReady()) return false;
        if (pending != null && discarded && pending.isDone()) {
            Future<float[]> finished = pending;
            pending = null;
            get(finished);
        }
        return pending == null;
    }

    public synchronized void submit(byte[] rgba) {
        if (!isIdle()) throw new IllegalStateException("PyTorch is still processing a frame");
        discarded = false;
        pending = executor.submit(() -> {
            long start = System.nanoTime();
            float[] rows = get(startup).callAttr("detect_rgba", (Object) rgba).toJava(float[].class);
            if (!reportedFrame) {
                Log.i("HoloCube", "HOLOCUBE_PT_CAMERA_FRAME: " + rows.length / 6 +
                    " detection rows in " + (System.nanoTime() - start) / 1000000 + " ms");
                reportedFrame = true;
            }
            return rows;
        });
    }

    /** True once the orientation checkpoint is loaded and its reference check passed. */
    public boolean isOrientationReady() { return orientationReady; }

    /** Why the orientation model is unavailable, or null while loading or when it is ready. */
    public String orientationError() { return orientationError; }

    /** One 640 x 640 RGBA image for the orientation model; the result is read with take(). */
    public synchronized void submitOrientation(byte[] rgba) {
        if (!orientationReady) throw new IllegalStateException("The orientation model is not loaded");
        if (!isIdle()) throw new IllegalStateException("PyTorch is still processing a frame");
        discarded = false;
        pending = executor.submit(() -> {
            long start = System.nanoTime();
            float[] rows = get(startup).callAttr("detect_orientation_rgba", (Object) rgba).toJava(float[].class);
            if (!reportedOrientationFrame) {
                Log.i("HoloCube", "HOLOCUBE_ORIENTATION_CAMERA_FRAME: " + rows.length / 12 +
                    " orientation rows in " + (System.nanoTime() - start) / 1000000 + " ms");
                reportedOrientationFrame = true;
            }
            return rows;
        });
    }

    public synchronized boolean isDone() { return pending != null && pending.isDone(); }

    public synchronized float[] take() {
        if (!isDone()) throw new IllegalStateException("The detection result is not ready");
        Future<float[]> finished = pending;
        pending = null;
        return get(finished);
    }

    public synchronized void discard() { discarded = true; }

    public synchronized void close() {
        closed = true;
        discarded = true;
        executor.shutdown(); // Finish native work without blocking Unity's render thread.
    }

    private static <T> T get(Future<T> future) {
        try { return future.get(); }
        catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("PyTorch worker interrupted", error);
        }
        catch (Exception error) {
            Throwable cause = error.getCause() == null ? error : error.getCause();
            throw new IllegalStateException("PyTorch: " + cause.getMessage(), cause);
        }
    }
}
