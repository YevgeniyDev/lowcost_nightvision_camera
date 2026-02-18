using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Text;

public class MJPEGHandler : DownloadHandlerScript
{
    private enum State
    {
        SearchingForBoundary,
        ReadingHeaders,
        ReadingImage
    }

    private State state = State.SearchingForBoundary;

    private string boundary = "--boundarydonotcross";
    private byte[] boundaryBytes;
    private StringBuilder headerBuilder = new StringBuilder();

    private int contentLength = 0;
    private int imageBytesRead = 0;
    private byte[] imageBuffer = new byte[1024 * 1024]; // 1 MB buffer

    public delegate void FrameEvent(byte[] frame);
    public event FrameEvent OnFrameComplete;

    public MJPEGHandler(byte[] buffer) : base(buffer)
    {
        boundaryBytes = Encoding.ASCII.GetBytes(boundary);
    }

    protected override bool ReceiveData(byte[] data, int dataLength)
    {
        int i = 0;

        while (i < dataLength)
        {
            switch (state)
            {
                case State.SearchingForBoundary:
                    if (MatchBoundary(data, i, dataLength))
                    {
                        state = State.ReadingHeaders;
                        headerBuilder.Length = 0;
                        i += boundaryBytes.Length;
                    }
                    else
                    {
                        i++;
                    }
                    break;

                case State.ReadingHeaders:
                    headerBuilder.Append((char)data[i]);

                    // Look for header termination (\r\n\r\n)
                    if (headerBuilder.Length >= 4 &&
                        headerBuilder.ToString().EndsWith("\r\n\r\n"))
                    {
                        ParseHeaders(headerBuilder.ToString());
                        state = State.ReadingImage;
                        imageBytesRead = 0;
                    }

                    i++;
                    break;

                case State.ReadingImage:
                    int remaining = contentLength - imageBytesRead;
                    int copy = Math.Min(remaining, dataLength - i);

                    Buffer.BlockCopy(data, i, imageBuffer, imageBytesRead, copy);
                    imageBytesRead += copy;
                    i += copy;

                    if (imageBytesRead >= contentLength)
                    {
                        byte[] jpeg = new byte[contentLength];
                        Buffer.BlockCopy(imageBuffer, 0, jpeg, 0, contentLength);

                        OnFrameComplete?.Invoke(jpeg);

                        state = State.SearchingForBoundary;
                    }
                    break;
            }
        }

        return true;
    }

    private bool MatchBoundary(byte[] data, int index, int length)
    {
        if (index + boundaryBytes.Length >= length) return false;

        for (int j = 0; j < boundaryBytes.Length; j++)
        {
            if (data[index + j] != boundaryBytes[j])
                return false;
        }

        return true;
    }

    private void ParseHeaders(string headers)
    {
        contentLength = 0;

        string[] lines = headers.Split('\n');

        foreach (string line in lines)
        {
            if (line.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase))
            {
                string value = line.Split(':')[1].Trim();
                contentLength = int.Parse(value);
            }
        }

        if (contentLength <= 0)
            Debug.LogError("ERROR: Invalid Content-Length in MJPEG stream");
    }
}
