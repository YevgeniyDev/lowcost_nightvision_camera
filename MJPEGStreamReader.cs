using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;

public class MJPEGStreamReader : MonoBehaviour
{
    public string streamURL = "http://10.101.41.111:8080/";
    public RawImage outputImage;

    private Texture2D tex;
    private UnityWebRequest request;

    void Start()
    {
        tex = new Texture2D(2, 2);

        byte[] buffer = new byte[4096];
        var handler = new MJPEGHandler(buffer);
        handler.OnFrameComplete += OnFrameReady;

        request = new UnityWebRequest(streamURL);
        request.downloadHandler = handler;
        request.disposeDownloadHandlerOnDispose = true;
        request.SendWebRequest();
    }

    void OnFrameReady(byte[] jpeg)
    {
        tex.LoadImage(jpeg);
        outputImage.texture = tex;
    }

    void OnDestroy()
    {
        if (request != null)
            request.Abort();
    }
}

