#pragma once

#include <d3d11.h>
//#include <d3dx11.h>

#include <string>
#include "DeviceHandler.h"

using namespace std;

class ShaderSet
{
public:
	ID3DBlob* m_vsData;
	ID3D11VertexShader* m_vs;

	ID3DBlob* m_psData;
	ID3D11PixelShader* m_ps;

private:
	DeviceHandler* m_deviceHandler;

public:
	ShaderSet( DeviceHandler* p_deviceHandler );
	~ShaderSet();

	void createSet( string p_filePath, string p_vsEntry, string p_psEntry );
private:

	void createVs();
	void createPs();
	ID3DBlob* readShader( const string &p_sourceFilePath );
	void ShaderSet::compileShader( const string &p_sourceFile,
		const string &p_entryPoint, const string &p_profile,
		ID3DBlob** out_blob );
};

