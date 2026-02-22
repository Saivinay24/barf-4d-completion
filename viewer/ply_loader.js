// viewer/ply_loader.js
async function loadPLY(url) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();

    // Parse PLY format (header + binary data)
    const text = new TextDecoder().decode(buffer.slice(0, 1000));
    const headerEnd = text.indexOf('end_header') + 'end_header'.length + 1;

    // Read vertex count from header
    const vertexCountMatch = text.match(/element vertex (\d+)/);
    const vertexCount = parseInt(vertexCountMatch[1]);

    // Parse binary vertex data (x, y, z, r, g, b)
    const dataView = new DataView(buffer, headerEnd);
    const points = [];
    const colors = [];

    for (let i = 0; i < vertexCount; i++) {
        const offset = i * 24; // float32 for xyz, uint8 for rgb
        points.push(
            dataView.getFloat32(offset, true),
            dataView.getFloat32(offset + 4, true),
            dataView.getFloat32(offset + 8, true)
        );
        colors.push(
            dataView.getUint8(offset + 12) / 255,
            dataView.getUint8(offset + 13) / 255,
            dataView.getUint8(offset + 14) / 255
        );
    }

    return { points, colors };
}
