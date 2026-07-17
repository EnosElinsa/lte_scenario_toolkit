(() => {
  const L = window.L;
  if (!L || typeof L.geoJSON !== "function") {
    throw new Error("Leaflet must load before the station dot extension");
  }
  if (typeof L.stationDots === "function") return;

  L.stationDots = (data, options = {}) => {
    const geoJsonOptions = { ...options };
    const dotStyle = {
      radius: 2.5,
      stroke: true,
      color: "#0b5f8a",
      weight: 1,
      opacity: 0.75,
      fillColor: "#4f93c8",
      fillOpacity: 0.55,
      interactive: true,
      bubblingMouseEvents: false,
      ...(geoJsonOptions.dotStyle || {}),
    };
    delete geoJsonOptions.dotStyle;
    geoJsonOptions.pointToLayer = (_feature, latlng) =>
      L.circleMarker(latlng, dotStyle);
    return L.geoJSON(data, geoJsonOptions);
  };
})();
