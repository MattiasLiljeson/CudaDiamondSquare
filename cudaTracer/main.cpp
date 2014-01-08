#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3DCompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

#include "DeviceHandler.h"
#include "cudaUtils.h"

#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include "TextureRenderer.h"
#include "globals.h"

#include<D3D11SDKLayers.h>
#include "D3DDebugger.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
				   LPSTR lpCmdLine, int nCmdShow)
{
	int wndWidth = 1280;
	int wndHeight = 720;
	DeviceHandler* deviceHandler = new DeviceHandler( hInstance, 700, 700);
	D3DDebugger d3dDbg(deviceHandler->getDevice());
	TextureRenderer* texRender = new TextureRenderer(deviceHandler, PIC_WIDTH, PIC_HEIGHT );

	MSG msg = {0};

	// Hard coded for now
	float dt = 1.0f/60.0f;

	while( msg.message != WM_QUIT ){
		if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) ){
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		} else {
			//timer.tick();
			//update( timer.getDt() );
			//renderer->draw();
			//renderer->setWindowTitle( timer.getDt() );

			texRender->update(dt);
			texRender->draw();
		}
	}

	delete texRender;
	delete deviceHandler;

	d3dDbg.reportLiveDeviceObjects();

	return 0;

}