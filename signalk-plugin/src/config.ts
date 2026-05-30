// Plugin configuration: TypeScript type + JSON Schema (for the SignalK admin UI).

export interface CameraOverride {
  name: string;
  hfovDeg: number;
  heightM: number;
  bearingOffsetDeg: number;
}

export interface PluginConfig {
  containerUrl: string;
  // Feature toggles
  enableVisualRadar: boolean;
  enableAisFusion: boolean;
  enableMob: boolean;
  enableCollision: boolean;
  enableAisBlips: boolean; // project visual targets as synthetic vessels.* (default off)
  enableContextControl: boolean;
  // Thresholds
  minConfidence: number;
  minRangeConfidence: number; // gate georeferencing
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
  enableVisualRadar: true,
  enableAisFusion: true,
  enableMob: true,
  enableCollision: true,
  enableAisBlips: false,
  enableContextControl: true,
  minConfidence: 0.4,
  minRangeConfidence: 0.3,
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
      enableVisualRadar: { type: 'boolean', title: 'Publish visual-radar targets (vision.targets.*)', default: true },
      enableAisFusion: { type: 'boolean', title: 'Fuse with AIS / detect dark targets', default: true },
      enableMob: { type: 'boolean', title: 'Man-overboard detection (notifications.mob)', default: true },
      enableCollision: { type: 'boolean', title: 'Collision risk (CPA/TCPA)', default: true },
      enableAisBlips: {
        type: 'boolean',
        title: 'Project visual targets as synthetic AIS vessels (advanced)',
        description: 'Renders blips on chartplotters as vessels.* with a VIS- name prefix. Off by default to avoid confusion with real AIS.',
        default: false,
      },
      enableContextControl: { type: 'boolean', title: 'Context-aware camera/model control', default: true },
      minConfidence: { type: 'number', title: 'Minimum detection confidence', default: 0.4, minimum: 0, maximum: 1 },
      minRangeConfidence: { type: 'number', title: 'Minimum range confidence to georeference', default: 0.3, minimum: 0, maximum: 1 },
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
      'enableVisualRadar',
      'enableAisFusion',
      'enableMob',
      'enableCollision',
      'enableContextControl',
      'enableAisBlips',
      '*',
    ],
  };
}

export function withDefaults(partial: Partial<PluginConfig> | undefined): PluginConfig {
  return { ...DEFAULT_CONFIG, ...(partial || {}) };
}
