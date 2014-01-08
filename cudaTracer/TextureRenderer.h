#pragma once

#include "DeviceHandler.h"
//#include "kernel.cuh"
#include "TextureSet.h"
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include "ShaderSet.h"

// Forward declarations
class DeviceHandler;

class TextureRenderer
{
private:
	DeviceHandler* m_deviceHandler;

	ID3D11InputLayout* m_inputLayout;
	TextureSet m_textureSet;
	ShaderSet* m_shaderSet;
	ID3D11Buffer* m_vertexBuffer;
	int m_texWidth;
	int m_texHeight;

	// Rasterizer states
	ID3D11RasterizerState *m_rsDefault;		// the default rasterizer state
	ID3D11RasterizerState *m_rsWireframe;	//Debug rasterizer

public:
	TextureRenderer( DeviceHandler* p_deviceHandler, int p_texWidth, int p_texHeight );
	~TextureRenderer();

	void update( float p_dt );
	void draw();

private:
	void initTexture();
	void initShaders();
	void initInputLayout();
	void initQuad();
	void initStates();
	void initInterop();
};

