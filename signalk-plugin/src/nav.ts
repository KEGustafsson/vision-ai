// Read own-ship navigation state from SignalK. headingTrue/cog are radians and
// sog is m/s in the SignalK model, so no conversion is needed.

import { ServerApp } from './skapp';
import { OwnShip, LatLon } from './types';

function num(v: any): number | null {
  return typeof v === 'number' && isFinite(v) ? v : null;
}

function pos(v: any): LatLon | null {
  if (v && typeof v.latitude === 'number' && typeof v.longitude === 'number') {
    return { latitude: v.latitude, longitude: v.longitude };
  }
  return null;
}

export function readOwnShip(app: ServerApp): OwnShip {
  return {
    position: pos(app.getSelfPath('navigation.position')?.value ?? app.getSelfPath('navigation.position')),
    headingTrue: num(app.getSelfPath('navigation.headingTrue')?.value ?? app.getSelfPath('navigation.headingTrue')),
    sog: num(app.getSelfPath('navigation.speedOverGround')?.value ?? app.getSelfPath('navigation.speedOverGround')),
    cog: num(app.getSelfPath('navigation.courseOverGroundTrue')?.value ?? app.getSelfPath('navigation.courseOverGroundTrue')),
  };
}
