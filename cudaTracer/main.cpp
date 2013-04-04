#include "DeviceHandler.h"
#include "cudaUtils.h"

#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include "TextureRenderer.h"
#include "globals.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
	LPSTR lpCmdLine, int nCmdShow)
{
//int main()
//{
	//_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	int wndWidth = 1280;
	int wndHeight = 720;
	DeviceHandler* deviceHandler = new DeviceHandler( 0L, wndWidth, wndHeight);
	deviceHandler->presentFrame();
	
	TextureRenderer* texRender = new TextureRenderer(deviceHandler, PIC_WIDTH, PIC_HEIGHT ); 

	MSG msg = {0};
	
	// Hard coded for now
	float dt = 1.0f/60.0f;

	while(msg.message != WM_QUIT)
	{
		if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			//timer.tick();
			//update( timer.getDt() );
			//renderer->draw();
			//renderer->setWindowTitle( timer.getDt() );

			texRender->update(dt);
			texRender->draw();
		}
	}

	add();

	return 0;
}