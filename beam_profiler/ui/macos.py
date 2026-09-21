"""Native close-button support for OpenCV's Cocoa viewer window."""
from AppKit import NSApplication, NSWindowCloseButton, NSWindowStyleMaskClosable
from Foundation import NSObject
import objc


class ProfilerCloseHandler(NSObject):
    @objc.python_method
    def configure(self, callback):
        self.callback = callback

    def closeProfiler_(self, sender):
        # Let the application loop release the camera and destroy its windows.
        # Directly closing Cocoa's window would bypass OpenCV's bookkeeping.
        self.callback()


def attach_close_button(title, callback):
    for window in NSApplication.sharedApplication().windows():
        if str(window.title()) == title:
            window.setStyleMask_(window.styleMask() | NSWindowStyleMaskClosable)
            button = window.standardWindowButton_(NSWindowCloseButton)
            handler = ProfilerCloseHandler.alloc().init()
            handler.configure(callback)
            button.setTarget_(handler)
            button.setAction_('closeProfiler:')
            button.setEnabled_(True)
            return handler  # Cocoa target is weak; retain for the viewer lifetime.
    raise RuntimeError('Cannot locate the beam-profiler Cocoa window')
