// Plugin configuration: TypeScript type + JSON Schema (for the SignalK admin UI).

export interface CameraOverride {
  name: string;
  hfovDeg: number;
  heightM: number;
  bearingOffsetDeg: number;
}

export interface PluginConfig {
  containerUrl: string;
  // Master on/off for detection in the container. Persisted default; the captain
  // webapp can flip it live (the live state is re-synced to the container).
  enableDetection: boolean;
  // Computation toggles — run the analysis and publish its data paths.
  enableVisualRadar: boolean; // publish vision.targets.* (data only, no alert)
  enableAisFusion: boolean; // AIS correlation + dark-target detection + vision.fusion.*
  enableCollision: boolean; // CPA/TCPA + threatLevel on vision.targets.*
  // Notification toggles — raise/clear the SignalK alert. Collision and dark
  // target require their computation (above) to be on.
  enableMob: boolean; // notifications.mob (no separate computation)
  notifyCollision: boolean; // notifications.vision.collision.*
  notifyDarkTarget: boolean; // notifications.vision.darkTarget.*
  enableAisBlips: boolean; // project visual targets as synthetic vessels.* (default off)
  enableContextControl: boolean;
  detectClasses: string[]; // object types to surface (person | vessel | buoy); empty => all
  // Thresholds
  minConfidence: number;
  minTargetRangeM: number; // drop any detection closer than this (own-hull / very-near clutter); 0 => off
  maxTargets: number; // maximum detections/tracks kept per frame in the container
  minRangeConfidence: number; // gate georeferencing
  ownAisMinRangeM: number; // ignore AIS contacts this close to own-ship
  darkTargetRangeM: number;
  correlationBearingDeg: number;
  correlationRangeFrac: number; // tolerance as fraction of range
  collisionTcpaS: number; // warn threshold
  collisionAlarmTcpaS: number; // alarm threshold
  collisionCpaM: number;
  mobMinConfidence: number;
  mobPersistFrames: number;
  underwaySogMs: number; // SOG above which we consider "underway"
  trackTimeoutS: number; // age out a visual track after this idle time
  processIntervalMs: number; // cadence of the fusion/CPA/notify/publish cycle
}

export const DEFAULT_CONFIG: PluginConfig = {
  containerUrl: 'http://localhost:7000',
  enableDetection: true,
  enableVisualRadar: true,
  enableAisFusion: true,
  enableCollision: true,
  enableMob: true,
  notifyCollision: true,
  notifyDarkTarget: true,
  enableAisBlips: false,
  enableContextControl: true,
  detectClasses: ['person', 'vessel', 'buoy'],
  minConfidence: 0.4,
  minTargetRangeM: 8,
  maxTargets: 20,
  minRangeConfidence: 0.3,
  ownAisMinRangeM: 25,
  darkTargetRangeM: 800,
  correlationBearingDeg: 8,
  correlationRangeFrac: 0.4,
  collisionTcpaS: 600,
  collisionAlarmTcpaS: 180,
  collisionCpaM: 100,
  mobMinConfidence: 0.5,
  mobPersistFrames: 3,
  underwaySogMs: 1.0,
  trackTimeoutS: 5,
  processIntervalMs: 1000,
};

export function schema(): object {
  return {
    type: 'object',
    title: 'Marine Vision-AI',
    description:
      'Controls the YOLOv8 vision container and turns detections into a ' +
      'georeferenced "visual radar", AIS fusion, MOB and collision alerts.',
    properties: {
      containerUrl: {
        type: 'string',
        title: 'Vision container base URL',
        default: DEFAULT_CONFIG.containerUrl,
      },
      enableDetection: {
        type: 'boolean',
        title: 'Enable detection (master on/off)',
        description: 'When off, the vision container keeps running but releases the cameras and stops all inference. Can also be toggled live from the captain webapp.',
        default: true,
      },
      // --- Computation: run the analysis and publish its data paths. ---
      enableVisualRadar: {
        type: 'boolean',
        title: 'Compute: visual-radar targets',
        description: 'Publishes each tracked detection under vision.targets.<camera>.<id>.* (bearing, range, position…). Data only — never raises a notification.',
        default: true,
      },
      enableCollision: {
        type: 'boolean',
        title: 'Compute: collision (CPA/TCPA)',
        description: 'Computes CPA/TCPA and threat level for each target (published on vision.targets.*). Required for the collision notification below.',
        default: true,
      },
      enableAisFusion: {
        type: 'boolean',
        title: 'Compute: AIS fusion / dark targets',
        description: 'Correlates visual targets with AIS contacts and flags non-AIS ("dark") targets; feeds the vision.fusion.* counts. Required for the dark-target notification below.',
        default: true,
      },
      // --- Notifications: raise/clear the SignalK alert (cleared when resolved). ---
      enableMob: {
        type: 'boolean',
        title: 'Notify: man overboard',
        description: 'Raises notifications.mob (state: emergency, visual+sound) when a person-in-water persists. Requires "person" in the detected object types below.',
        default: true,
      },
      notifyCollision: {
        type: 'boolean',
        title: 'Notify: collision risk',
        description: 'Raises notifications.vision.collision.<track> (state: warn, then alarm; visual+sound) for risky approaches. Needs "Compute: collision" on. Turn this off to keep CPA data on the radar without an audible alarm.',
        default: true,
      },
      notifyDarkTarget: {
        type: 'boolean',
        title: 'Notify: dark target (non-AIS)',
        description: 'Raises notifications.vision.darkTarget.<track> (state: alert, visual) for targets with no AIS match. Needs "Compute: AIS fusion" on.',
        default: true,
      },
      enableAisBlips: {
        type: 'boolean',
        title: 'Project visual targets as synthetic AIS vessels (advanced)',
        description: 'Renders blips on chartplotters as vessels.* with a VIS- name prefix. Off by default to avoid confusion with real AIS.',
        default: false,
      },
      enableContextControl: {
        type: 'boolean',
        title: 'Context-aware camera/model control',
        description:
          'Automatically adapts the active camera and detection sensitivity to the situation. ' +
          'By speed: underway (SOG ≥ "Underway SOG threshold") watches the forward camera; ' +
          'slow/stopped switches to the aft camera for docking. By time of day: at night ' +
          '(21:00–06:00) the confidence threshold is lowered by 0.1 (floor 0.25) to catch dimmer ' +
          'targets. When off, the camera stays fixed (no auto-switch) and confidence stays at ' +
          '"Minimum detection confidence" — pick the camera manually from the captain webapp.',
        default: true,
      },
      detectClasses: {
        type: 'array',
        title: 'Object types to detect',
        description:
          'Which detections to surface, draw on the video, and alert on. ' +
          'Person / Vessel / Buoy work with the stock COCO model; Debris / ' +
          'Kayak / Log require the forward-watch model; Boat / Sailboat / ' +
          'Speedboat / Warship require the marine-surveillance model (see ' +
          'detector.model in deepstream.yaml). The plugin warns if you pick ' +
          'labels the active model cannot produce. Note: man-overboard detection ' +
          'requires "person", which only the COCO model has. Leave all unchecked ' +
          'to detect everything.',
        items: {
          type: 'string',
          enum: ['person', 'vessel', 'buoy', 'debris', 'kayak', 'log',
                 'boat', 'sailboat', 'speedboat', 'warship'],
        },
        uniqueItems: true,
        default: ['person', 'vessel', 'buoy'],
      },
      minConfidence: { type: 'number', title: 'Minimum detection confidence', default: 0.4, minimum: 0, maximum: 1 },
      maxTargets: {
        type: 'number',
        title: 'Maximum targets per frame',
        description: 'Caps YOLO detections before tracking/event generation. Lower values reduce workload in busy scenes.',
        default: 20,
        minimum: 1,
        maximum: 300,
      },
      minTargetRangeM: {
        type: 'number',
        title: 'Ignore detections closer than (m)',
        description: 'Drops any detected object whose estimated range is below this — own-hull artifacts and very-near clutter that swamp the frame and create phantom alerts. Filtered in the vision container, so too-close objects are removed from BOTH the target list and the annotated video overlay. Person is exempt (man-overboard must be seen up close); detections with no range estimate are kept. Set 0 to disable.',
        default: 8,
        minimum: 0,
      },
      minRangeConfidence: { type: 'number', title: 'Minimum range confidence to georeference', default: 0.3, minimum: 0, maximum: 1 },
      ownAisMinRangeM: {
        type: 'number',
        title: 'Ignore AIS contacts closer than (m)',
        description: 'Filters duplicate own-ship AIS entries before visual/AIS correlation. Set 0 to disable.',
        default: 25,
        minimum: 0,
      },
      darkTargetRangeM: { type: 'number', title: 'Dark-target alert range (m)', default: 800 },
      correlationBearingDeg: { type: 'number', title: 'AIS correlation bearing tolerance (deg)', default: 8 },
      correlationRangeFrac: { type: 'number', title: 'AIS correlation range tolerance (fraction)', default: 0.4 },
      collisionTcpaS: { type: 'number', title: 'Collision warn TCPA (s)', default: 600 },
      collisionAlarmTcpaS: { type: 'number', title: 'Collision alarm TCPA (s)', default: 180 },
      collisionCpaM: { type: 'number', title: 'Collision CPA threshold (m)', default: 100 },
      mobMinConfidence: { type: 'number', title: 'MOB minimum confidence', default: 0.5, minimum: 0, maximum: 1 },
      mobPersistFrames: { type: 'number', title: 'MOB persistence (frames)', default: 3 },
      underwaySogMs: { type: 'number', title: 'Underway SOG threshold (m/s)', default: 1.0 },
      trackTimeoutS: { type: 'number', title: 'Track age-out timeout (s)', default: 5 },
      processIntervalMs: { type: 'number', title: 'Processing cadence (ms)', default: 1000, minimum: 200 },
    },
  };
}

export function uiSchema(): object {
  return {
    'ui:order': [
      'containerUrl',
      'enableDetection',
      // Computation toggles grouped first…
      'enableVisualRadar',
      'enableCollision',
      'enableAisFusion',
      // …then the notification toggles.
      'enableMob',
      'notifyCollision',
      'notifyDarkTarget',
      'enableContextControl',
      'enableAisBlips',
      'detectClasses',
      'minConfidence',
      'minTargetRangeM',
      'maxTargets',
      'ownAisMinRangeM',
      '*',
    ],
  };
}

export function withDefaults(partial: Partial<PluginConfig> | undefined): PluginConfig {
  return { ...DEFAULT_CONFIG, ...(partial || {}) };
}
