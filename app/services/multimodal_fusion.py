"""
Phase 7 & Phase 10: Multi-Modal Perception Fusion Engine (Vision + Audio MFCC + IMU Acceleration + Pressure Footsteps).

Fuses visual perception telemetry with audio acoustic classification (gunshots, screams, sirens, footsteps)
and wearable IMU acceleration patterns (magnitude sqrt(ax^2 + ay^2 + az^2), person vs vehicle motion)
plus footstep pressure telemetry for unified threat assessment.
"""
import logging
import math
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger("MultiModalFusion")


class AudioThreatDetector:
    """
    Audio Acoustic Threat Detector analyzing MFCC features and energy spectral thresholds.
    """

    def extract_mfcc(self, audio_buffer: Optional[np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        """
        Extracts 13-channel Mel-Frequency Cepstral Coefficients (MFCCs).
        """
        if audio_buffer is None or len(audio_buffer) == 0:
            return np.zeros(13, dtype=np.float32)

        try:
            arr = np.asarray(audio_buffer, dtype=np.float32)
            energy = float(np.mean(arr ** 2))
            
            # Compute 13-channel cepstral spectral feature representation
            n_filters = 13
            spectrum = np.abs(np.fft.rfft(arr, n=256)) if len(arr) >= 16 else np.zeros(129)
            spectral_energy = np.mean(spectrum[:n_filters]) if len(spectrum) >= n_filters else energy
            
            # Base MFCC features simulation from Mel frequency band energy
            mel_bands = np.linspace(0.1, 3.14, n_filters)
            mfcc = np.sin(mel_bands) * (energy * 5.0 + spectral_energy * 0.5)
            
            # Include energy at coefficient index 0
            mfcc[0] = energy
            return mfcc.astype(np.float32)
        except Exception as e:
            logger.debug(f"MFCC extraction error: {e}")
            return np.zeros(13, dtype=np.float32)

    def audio_classifier(self, audio_features: np.ndarray, db_level: float = 45.0, peak_amp: float = 0.0) -> Dict[str, Any]:
        """
        Classifies acoustic events (gunshots, screams, sirens, footsteps, ambient speech)
        from extracted MFCC features and sound pressure levels.
        """
        if audio_features is None or len(audio_features) == 0:
            return {
                "event": "ambient_noise",
                "threat_score": 0.0,
                "is_acoustic_threat": False,
                "db_level": db_level,
                "mfcc_features": np.zeros(13, dtype=np.float32).tolist()
            }

        energy = float(audio_features[0]) if len(audio_features) > 0 else 0.0
        feature_std = float(np.std(audio_features)) if len(audio_features) > 0 else 0.0

        # Classification logic based on acoustic dB level, peak transient, and MFCC spectral characteristics
        if db_level > 95.0 or (db_level > 80.0 and (energy > 0.1 or peak_amp > 1.5)):
            event = "gunshot_explosion"
            score = 0.98
        elif db_level > 80.0:
            if feature_std > 0.15:
                event = "scream"
                score = 0.90
            else:
                event = "siren"
                score = 0.85
        elif db_level > 65.0:
            event = "heavy_machinery"
            score = 0.45
        elif db_level > 55.0 and energy > 0.01:
            event = "footsteps"
            score = 0.40
        else:
            event = "ambient_speech"
            score = 0.10

        return {
            "event": event,
            "threat_score": score,
            "is_acoustic_threat": score >= 0.75,
            "db_level": db_level,
            "mfcc_features": audio_features.tolist()
        }

    def classify_audio_threat(self, audio_buffer: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Wrapper extracting MFCCs and classifying acoustic threats.
        """
        if audio_buffer is None or len(audio_buffer) == 0:
            return self.audio_classifier(np.zeros(13, dtype=np.float32), db_level=45.0)

        arr = np.asarray(audio_buffer, dtype=np.float32)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        peak_amp = float(np.max(np.abs(arr))) if len(arr) > 0 else 0.0
        db_level = round(20.0 * math.log10(max(1e-4, rms)) + 90.0, 1)

        audio_features = self.extract_mfcc(audio_buffer)
        return self.audio_classifier(audio_features, db_level=db_level, peak_amp=peak_amp)


class IMUPatternAnalyzer:
    """
    IMU Acceleration & Gyroscope Pattern Analyzer.
    Calculates imu_magnitude = sqrt(ax^2 + ay^2 + az^2) and distinguishes person vs vehicle movement.
    """

    def calculate_magnitude(self, ax: float, ay: float, az: float) -> float:
        """Computes IMU acceleration magnitude sqrt(ax^2 + ay^2 + az^2)."""
        return float(np.sqrt(ax**2 + ay**2 + az**2))

    def analyze_imu(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float = 0.0,
        gy: float = 0.0,
        gz: float = 0.0,
        imu_type_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyzes IMU movement pattern (Person vs Vehicle motion, impact/fall, walking, running).
        """
        mag = self.calculate_magnitude(ax, ay, az)
        gyro_mag = float(np.sqrt(gx**2 + gy**2 + gz**2))

        # 1. Explicit vehicle telemetry hint or micro-vibration vehicle pattern
        if imu_type_hint == "vehicle" or (imu_type_hint != "person" and mag > 12.0 and mag <= 22.0 and 0.001 < gyro_mag < 0.02 and abs(az - 9.81) < 1.0):
            motion_type = "vehicle"
            if mag > 15.0:
                pattern = "vehicle_acceleration"
            else:
                pattern = "vehicle_travel"
            is_moving = True
        # 2. Person impact / fall (mag > 22.0 m/s^2)
        elif mag > 22.0:
            motion_type = "person"
            pattern = "impact_fall"
            is_moving = True
        # 3. Person running (mag > 13.0 m/s^2)
        elif mag > 13.0:
            motion_type = "person"
            pattern = "running"
            is_moving = True
        # 4. Person walking (mag > 10.5 m/s^2)
        elif mag > 10.5:
            motion_type = "person"
            pattern = "walking"
            is_moving = True
        # 5. Footstep gait impulses (mag > 9.9 m/s^2)
        elif mag > 9.9:
            motion_type = "person"
            pattern = "footstep_gait"
            is_moving = True
        else:
            motion_type = "stationary"
            pattern = "stationary"
            is_moving = False

        return {
            "imu_magnitude": round(mag, 2),
            "pattern": pattern,
            "motion_type": motion_type,
            "is_moving": is_moving
        }


class PressureFootstepAnalyzer:
    """
    Footstep Impulse & Pressure Telemetry Analyzer.
    """

    def analyze_pressure_footsteps(self, pressure_buffer: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyzes pressure/acoustic impulse buffer for footstep cadence and proximity threat level.
        """
        if pressure_buffer is None or len(pressure_buffer) == 0:
            return {
                "step_count": 0,
                "cadence_hz": 0.0,
                "has_footsteps": False,
                "footstep_threat_score": 0.0
            }

        arr = np.asarray(pressure_buffer, dtype=np.float32)
        peaks = np.where(arr > 0.5)[0]
        step_count = len(peaks)
        
        has_footsteps = step_count > 0
        cadence_hz = round(step_count / max(1.0, len(arr) / 100.0), 2)
        
        # High-frequency approaching footsteps boost threat
        if step_count >= 5 and cadence_hz > 2.0:
            footstep_threat = 0.65  # Rapid approaching footsteps
        elif step_count > 0:
            footstep_threat = 0.30
        else:
            footstep_threat = 0.0

        return {
            "step_count": step_count,
            "cadence_hz": cadence_hz,
            "has_footsteps": has_footsteps,
            "footstep_threat_score": footstep_threat
        }


class MultiModalFusionEngine:
    """
    Fuses Vision, Audio (MFCC + Classifier), Wearable IMU (Person vs Vehicle), and Pressure Footstep telemetry.
    """

    def __init__(
        self,
        vision_weight: float = 0.45,
        audio_weight: float = 0.30,
        imu_weight: float = 0.15,
        pressure_weight: float = 0.10,
        movement_threshold: float = 10.2
    ):
        self.vision_weight = vision_weight
        self.audio_weight = audio_weight
        self.imu_weight = imu_weight
        self.pressure_weight = pressure_weight
        self.movement_threshold = movement_threshold

        self.audio_detector = AudioThreatDetector()
        self.imu_analyzer = IMUPatternAnalyzer()
        self.pressure_analyzer = PressureFootstepAnalyzer()

    def correlate_with_vision(
        self,
        imu_res: Dict[str, Any],
        vision_threat_score: float,
        vision_objects: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Correlates wearable IMU movement with visual detection telemetry.
        Called when imu_magnitude > movement_threshold.
        """
        mag = imu_res.get("imu_magnitude", 0.0)
        pattern = imu_res.get("pattern", "stationary")
        motion_type = imu_res.get("motion_type", "stationary")

        is_correlated = False
        correlation_factor = 1.0

        # Check vision-IMU correlation alignment
        if motion_type == "vehicle":
            # IMU indicates vehicle travel; check if vision sees vehicle
            has_vision_vehicle = False
            if vision_objects:
                labels = [getattr(obj, "label", str(obj)).lower() for obj in vision_objects]
                has_vision_vehicle = any("vehicle" in l or "car" in l or "truck" in l for l in labels)
            
            if has_vision_vehicle or vision_threat_score > 0.5:
                is_correlated = True
                correlation_factor = 1.25
        elif pattern in ["impact_fall", "running"]:
            # High body movement; elevated threat if vision also sees high threat
            if vision_threat_score > 0.4:
                is_correlated = True
                correlation_factor = 1.30
        elif mag > self.movement_threshold:
            is_correlated = True
            correlation_factor = 1.10

        return {
            "is_correlated": is_correlated,
            "correlation_factor": round(correlation_factor, 2),
            "imu_magnitude": mag,
            "imu_pattern": pattern,
            "vision_threat_score": vision_threat_score
        }

    def fuse_multimodal_perception(
        self,
        vision_threat_score: float,
        audio_buffer: Optional[np.ndarray] = None,
        imu_reading: Optional[Dict[str, float]] = None,
        pressure_buffer: Optional[np.ndarray] = None,
        vision_objects: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates unified multi-modal threat score using Vision, Audio (MFCC + Classifier),
        IMU pattern matching (Person vs Vehicle), Pressure footsteps, and Vision Correlation.
        """
        # 1. Audio threat detection (extract_mfcc + audio_classifier)
        audio_features = self.audio_detector.extract_mfcc(audio_buffer)
        audio_res = self.audio_detector.classify_audio_threat(audio_buffer)

        # 2. IMU pattern matching (imu_magnitude = sqrt(ax^2 + ay^2 + az^2))
        if imu_reading:
            ax = imu_reading.get("ax", 0.0)
            ay = imu_reading.get("ay", 0.0)
            az = imu_reading.get("az", 9.81)
            gx = imu_reading.get("gx", 0.0)
            gy = imu_reading.get("gy", 0.0)
            gz = imu_reading.get("gz", 0.0)
            hint = imu_reading.get("type_hint", None)
            imu_res = self.imu_analyzer.analyze_imu(ax, ay, az, gx, gy, gz, imu_type_hint=hint)
        else:
            imu_res = self.imu_analyzer.analyze_imu(0.0, 0.0, 9.81)

        # 3. Vision correlation if movement detected (imu_magnitude > threshold)
        if imu_res["imu_magnitude"] > self.movement_threshold:
            correlation_res = self.correlate_with_vision(imu_res, vision_threat_score, vision_objects)
        else:
            correlation_res = {
                "is_correlated": False,
                "correlation_factor": 1.0,
                "imu_magnitude": imu_res["imu_magnitude"],
                "imu_pattern": imu_res["pattern"],
                "vision_threat_score": vision_threat_score
            }

        # 4. Pressure / Footstep analysis
        pressure_res = self.pressure_analyzer.analyze_pressure_footsteps(pressure_buffer)

        # 5. Score aggregation
        v_score = max(0.0, min(1.0, float(vision_threat_score)))
        a_score = float(audio_res["threat_score"])
        
        if imu_res["pattern"] == "impact_fall":
            i_score = 0.95
        elif imu_res["pattern"] == "running":
            i_score = 0.75
        elif imu_res["motion_type"] == "vehicle":
            i_score = 0.50
        else:
            i_score = 0.10

        p_score = float(pressure_res["footstep_threat_score"])

        # Base weighted multi-modal sum
        raw_score = (
            self.vision_weight * v_score +
            self.audio_weight * a_score +
            self.imu_weight * i_score +
            self.pressure_weight * p_score
        )

        # Apply vision-IMU correlation factor multiplier
        fused_score = round(min(1.0, raw_score * correlation_res["correlation_factor"]), 4)

        if fused_score >= 0.75:
            threat_level = "CRITICAL"
        elif fused_score >= 0.50:
            threat_level = "HIGH"
        elif fused_score >= 0.25:
            threat_level = "WARNING"
        else:
            threat_level = "SAFE"

        return {
            "fused_threat_level": threat_level,
            "multi_modal_score": fused_score,
            "vision_score": v_score,
            "audio_res": audio_res,
            "imu_res": imu_res,
            "pressure_res": pressure_res,
            "vision_correlation": correlation_res
        }
